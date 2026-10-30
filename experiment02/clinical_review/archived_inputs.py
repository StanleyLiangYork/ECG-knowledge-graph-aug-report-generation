"""Build native clinical-review packages from verified historical PNGs without rerendering.

The original workflow and its frozen hash checks remain unchanged. This adapter
records the additional reconstruction provenance in the new review package.
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from common import atomic, read, sha, signature


def prepare_from_reconstruction(cfg, folders, destination, reconstruction_dir, **settings):
    """Use one verified rendering group for every selected generator.

    No model is loaded, no waveform is rerendered, and no native report provenance
    is rewritten. A mixed PULSE/modern review still cannot share one case image.
    """
    from common import assert_config
    from clinical_review.workflow import prepare
    from extensions import rendering_audit
    import queries

    assert_config(cfg)
    destination, archive = Path(destination), Path(reconstruction_dir)
    if destination.exists():
        raise FileExistsError('Existing clinical package preserved; resume it or select a new review version')
    if cfg['cohort'] not in ('internal_val', 'internal_test'):
        raise ValueError('The current historical reconstruction adapter supports internal Symile cohorts only')
    run = Path(cfg['run_dir'])
    packet = run / 'packets' / cfg['cohort'] / 'manifest.json'
    query_manifest = run / 'queries' / cfg['cohort'] / 'manifest.json'
    protocol_path, status_path = archive / 'protocol.json', archive / 'status.json'
    if not protocol_path.is_file() or not status_path.is_file():
        raise FileNotFoundError('Historical rendering protocol/status missing at ' + str(archive)
            + '. Select the completed BR05A modern reconstruction, not the experiment-level protocol.')
    reconstruction, status = read(protocol_path), read(status_path)
    if (not status.get('complete')
            or status.get('protocol_sha256') != sha(protocol_path)):
        raise ValueError('Historical rendering audit is incomplete or its protocol has changed')
    manifest = read(packet)
    if (reconstruction['cohort'] != cfg['cohort']
            or reconstruction['packet_manifest_sha256'] != sha(packet)
            or reconstruction['query_manifest_sha256'] != sha(query_manifest)
            or set(reconstruction['expected_sources']) != set(manifest['query_ids'])
            or status.get('total') != len(manifest['query_ids'])
            or status.get('reproduced') != len(manifest['query_ids'])):
        raise ValueError('Historical reconstructions do not cover this exact frozen query cohort')
    if reconstruction['helper_sha256'] != sha(rendering_audit.__file__):
        raise ValueError('Historical rendering helper differs from the completed audit; preserve the audited helper version')
    uncovered = set(folders) - set(reconstruction['sources'])
    if uncovered:
        raise ValueError('Generators absent from this rendering group: ' + ', '.join(sorted(uncovered))
            + '. Use llama_ecg and qwen38 with the modern archive; do not combine PULSE and modern images as one case image.')
    for model, folder in folders.items():
        folder = Path(folder).resolve()
        source = reconstruction['sources'][model]
        if (folder != Path(source['folder']).resolve()
                or sha(folder / 'protocol.json') != source['protocol']
                or sha(folder / 'reports.jsonl') != source['reports']):
            raise ValueError('Selected generation folder/provenance differs from the completed reconstruction: ' + model)

    verified = {}
    def provide(row):
        q = row['id']
        if q not in verified:
            if q not in reconstruction['expected_sources'] or Path(q).name != q:
                raise ValueError('Query absent from historical rendering audit')
            check_path, image = archive / 'checks' / (q + '.json'), archive / 'images' / (q + '.png')
            if not check_path.is_file() or not image.is_file():
                raise FileNotFoundError('Missing historical query PNG/check: ' + str(image))
            check, expected = read(check_path), reconstruction['expected_sources'][q]
            if (check['query_id'] != q
                    or check['protocol_sha256'] != signature(reconstruction)
                    or check['tensor_sha256'] != expected['tensor']
                    or check['png_sha256'] != expected['png']
                    or sha(image) != expected['png']
                    or rendering_audit.pixel_identity(image) != check['pixels']):
                raise ValueError('Historical image reconstruction proof failed: ' + q)
            verified[q] = (str(image), check['tensor_sha256'])
        return verified[q]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix='.historical_review_', dir=destination.parent))
    package = staging_root / 'package'
    try:
        # Native prepare still validates every selected model/arm against the
        # exact image file and tensor identity before copying anonymous assets.
        with patch.object(queries, 'image_provider', return_value=provide):
            result = prepare(cfg, folders, package, **settings)
        protocol = read(package / 'PRIVATE/protocol.json')
        extra_sources = [protocol_path, status_path, Path(__file__), Path(rendering_audit.__file__)]
        for q in sorted(verified):
            extra_sources.extend([archive / 'checks' / (q + '.json'), archive / 'images' / (q + '.png')])
        seen = {r['file'] for r in protocol['sources']}
        for source in extra_sources:
            source = source.resolve()
            if str(source) not in seen:
                protocol['sources'].append(dict(file=str(source), sha256=sha(source)))
                seen.add(str(source))
        protocol['image_source'] = dict(mode='verified_historical_png',
            reconstruction_dir=str(archive.resolve()), reconstruction_protocol_sha256=sha(protocol_path),
            adapter_sha256=sha(__file__), rendering_group_models=sorted(reconstruction['sources']),
            selected_models=list(folders), rendered_new_images=False,
            rule='Native report PNG and tensor hashes must match the verified historical image; no pixel/file identity waiver')
        atomic(package / 'PRIVATE/protocol.json', protocol)
        if destination.exists():
            raise FileExistsError('Another package now exists; it has been preserved')
        package.rename(destination)
        result.update(destination=str(destination), image_source='verified_historical_png', rendered_new_images=False)
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
