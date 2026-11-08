"""Codex CLI policy, isolated from simulator assets and shadow output."""
import base64,json,re,subprocess,time
from pathlib import Path
try:
    from .llm_rollout import INSTRUCTIONS,decision_schema,append
except ImportError:
    from llm_rollout import INSTRUCTIONS,decision_schema,append


class CodexBackend:
    def __init__(self,inputs,audit,container='savnce_codex_actor_v1',model='gpt-6.1-sol'):
        self.inputs=Path(inputs);self.audit=Path(audit);self.container=container;self.model=model;self.effort='low'
    def decide(self,summary,history,images,actions,evidence,feedback,call_tool=None):
        if call_tool is not None:raise ValueError('Shadow results may not be exposed to the policy')
        for i,image in enumerate(images):(self.inputs/f'image_{i}.png').write_bytes(base64.b64decode(image,validate=True))
        schema=decision_schema(summary['observation_id'],actions,evidence)
        (self.inputs/'schema.json').write_text(json.dumps(schema))
        payload={'current':summary,'past_steps_oldest_first':list(history),'available_evidence_ids':sorted(evidence),
                 'available_actions':list(actions),'feedback_on_previous_proposal':feedback}
        instructions=INSTRUCTIONS
        mode=summary.get('control',{}).get('mode')
        if mode in ('synchronous_segments_v1','stationary_listen_v1'):
            instructions=instructions.replace(
                'Choose exactly one primitive action.\nmove_forward advances 0.25 m, turn_left and turn_right rotate 15 degrees, stop ends the episode.',
                'Choose one bounded action segment. The coordinator repeats your primitive for up to '
                f"{summary['control']['max_native_steps']} native steps (.25s each), at most 1 simulated second. "
                'Each forward primitive advances .25m; each turn primitive rotates 15 degrees. '
                'A full four-step segment can move 1m or rotate 60 degrees. STOP is immediate. '
                'Collision, close/unknown forward depth or existing arrival evidence may interrupt early. '
                'The simulator is paused at current.control.latest_pose during this request. '
                'Current RGB-D and pose are latest; past audio belongs to its original observation poses. '
                'The stereo panel and current audio summary cover only the latest .25s, not a 1s average.')
            if mode=='stationary_listen_v1':
                instructions=instructions.replace(
                    'The stereo panel and current audio summary cover only the latest .25s, not a 1s average.',
                    'Before this request the coordinator collected four fresh .25s blocks at fixed position and orientation. '
                    'The stereo panel and current.stationary_audio summarize that full 1s. '
                    'current.audio is still the latest .25s; passive fits keep their original per-step semantics. '
                    'After movement the coordinator will again listen in place for 1s before requesting a new decision. '
                    'Listening advances simulation/audio time; waiting for your reply does not. '
                    'Do not treat the aggregate and its four constituent blocks as independent evidence.')
                instructions=instructions.replace('The first image is current RGB; the second is a stereo evidence panel computed from the current 0.25 s window.',
                    'The first image is latest RGB at the listening pose; the second summarizes the stationary 1s binaural window.')
        prompt=instructions+'\nYou are only selecting a robot action. Do not use tools, browse, inspect files, or request other agents. Return the final JSON matching the supplied schema.\n'+json.dumps(payload,allow_nan=False,separators=(',',':'))
        command=['docker','exec','-i',self.container,'codex','exec','--model',self.model,'--ephemeral','--skip-git-repo-check',
            '--sandbox','read-only','--json','--color','never','--output-schema','/inputs/schema.json','-C','/inputs']
        for i in range(len(images)):command+=['--image',f'/inputs/image_{i}.png']
        command+=['-']
        started=time.monotonic();result=subprocess.run(command,input=prompt,capture_output=True,text=True,timeout=150)
        events=[]
        for line in result.stdout.splitlines():
            try:events.append(json.loads(line))
            except ValueError:pass
        texts=[];usage={};item_types=[]
        for event in events:
            item=event.get('item',{});kind=item.get('type')
            if kind:item_types.append(kind)
            if kind in ('command_execution','mcp_tool_call','web_search','file_change','collab_tool_call'):
                raise RuntimeError('Codex attempted an undeclared tool; no navigation action executed')
            if kind=='agent_message':texts.append(item['text'])
            if event.get('type')=='turn.completed':usage=event.get('usage',{})
        safe_error=re.sub(r'\bsk-[A-Za-z0-9_\-]+','[REDACTED]',result.stderr[-1500:])
        append(self.audit,dict(observation_id=summary['observation_id'],model=self.model,auth_mode='api_key',
            returncode=result.returncode,latency_s=time.monotonic()-started,item_types=item_types,usage=usage,
            error=safe_error if result.returncode else None,shadow_visible=False,simulator_assets_mounted=False))
        if result.returncode!=0 or not texts:raise RuntimeError('Codex request failed; see codex_calls.jsonl')
        return json.loads(texts[-1]),{k:usage.get(k,0) for k in ('input_tokens','output_tokens')}
