"""Read-only localhost resource checks; never create reference coordinates."""
import json,hashlib,urllib.request,urllib.error,datetime
from pathlib import Path
from run_fixed_cohort_player_case_v3 import BASE,dump

def main():
    pilot=BASE/'identity_position_bundle/reference_pilot';runtime=json.loads((pilot/'server_runtime.json').read_text(encoding='utf-8'));url=runtime['url'];assert url.startswith('http://127.0.0.1:')
    data=json.load(urllib.request.urlopen(url+'api/session',timeout=5));assert set(data)=={'manifest','annotations'} and len(data['manifest']['items'])==13
    private=json.loads((pilot/'private_sampling_manifest.json').read_text(encoding='utf-8'));hashes={r['image_id']:r['image_sha256'] for r in private['items']}
    for item in data['manifest']['items']:
        assert set(item)=={'image_id','width','height','image_file'}
        with urllib.request.urlopen(url+'images/'+item['image_file'],timeout=5) as response:assert hashlib.sha256(response.read()).hexdigest()==hashes[item['image_id']]
    for route in ['private_sampling_manifest.json','private_sampling_protocol.json','../position_policy_observations.csv']:
        try:urllib.request.urlopen(url+route,timeout=5)
        except urllib.error.HTTPError as error:assert error.code==404
        else:raise AssertionError('Unexpected private route')
    report=dict(status='passed',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session_image_count=13,image_sha256_checks=13,private_routes_rejected=3,real_server_requests='GET only; no annotations written by test',reference_revision_at_check=data['annotations']['revision'],browser_rendering_verified=False)
    dump(pilot.parent/'live_http_verification.json',report);print(url);print(report)

if __name__=='__main__':main()
