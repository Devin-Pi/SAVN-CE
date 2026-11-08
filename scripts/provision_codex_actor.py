import json,os,pathlib,re,subprocess
if subprocess.run(['docker','inspect','savnce_codex_actor_v1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
 raise SystemExit('savnce_codex_actor_v1 already exists; reuse it, or explicitly remove your own stopped actor before reprovisioning')
root=pathlib.Path('/media/4tbssd/00_PROJECT_CODE/SAVN-CE/outputs/avn_candidate_shadow_worktree')
private=pathlib.Path.home()/'.config/savnce-codex-shadow';private.mkdir(parents=True,exist_ok=True);private.chmod(0o700)
auth=json.loads((pathlib.Path.home()/'.codex/auth.json').read_text());key=auth.get('OPENAI_API_KEY');assert key
base=re.search(r'(?m)^\s*base_url\s*=\s*"([^"]+)"',(pathlib.Path.home()/'.codex/config.toml').read_text()).group(1)
(private/'auth.json').write_text(json.dumps({'OPENAI_API_KEY':key}));(private/'auth.json').chmod(0o600)
config='model_provider="codex"\nmodel="gpt-6.1-sol"\nmodel_reasoning_effort="low"\ncli_auth_credentials_store="file"\nweb_search="disabled"\n[model_providers.codex]\nname="Existing user gateway"\nbase_url='+json.dumps(base)+'\nwire_api="responses"\nrequires_openai_auth=true\nrequest_max_retries=1\nstream_max_retries=1\n[features]\n'
for feature in ('shell_tool','multi_agent','plugins','apps','hooks','browser_use','computer_use','view_image','memories','skill_search','sleep_tool','goals','workspace_dependencies'):
 config+=feature+'=false\n'
(private/'config.toml').write_text(config);(private/'config.toml').chmod(0o600)
inputs=root/'outputs/codex_actor_inputs_20261108';inputs.mkdir(parents=True,exist_ok=True)
binary=pathlib.Path.home()/'.nvm/versions/node/v22.16.0/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'
command=['docker','run','-d','--name','savnce_codex_actor_v1','--read-only','--user',str(os.getuid())+':'+str(os.getgid()),'--cap-drop=ALL','--security-opt=no-new-privileges',
 '--tmpfs','/tmp:rw,exec,size=256m','--tmpfs','/codex_home:rw,mode=700,size=64m,uid='+str(os.getuid())+',gid='+str(os.getgid()),'-e','CODEX_HOME=/codex_home',
 '-v',str(binary)+':/usr/local/bin/codex:ro','-v',str(private/'auth.json')+':/codex_home/auth.json:ro',
 '-v',str(private/'config.toml')+':/codex_home/config.toml:ro','-v','/etc/ssl/certs:/etc/ssl/certs:ro',
 '-v',str(inputs)+':/inputs:ro','--workdir','/inputs','ubuntu:latest','sleep','infinity']
p=subprocess.run(command,capture_output=True,text=True);print(json.dumps(dict(returncode=p.returncode,container_id=p.stdout.strip(),error=p.stderr.strip(),model='gpt-6.1-sol',credential_values_logged=False,scene_assets_mounted=False)))
