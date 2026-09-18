#!/usr/bin/env python3
"""FlyGym 2.1 CPU physics feasibility only; no learned or language-model controller.

Run in isolated Python 3.12 with flygym==2.1.0. Adapted API usage from
https://neuromechfly.org/tutorials/4c_hybrid_controller/ (Apache-2.0 FlyGym).
No renderer, Warp, training, or biological validation is involved.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata as md, json, os, platform, sys, time, traceback

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/robot-simulator-smoke'))
    parser.add_argument('--seconds',type=float,default=.1)
    args=parser.parse_args()
    if not .1 <= args.seconds <= .5: parser.error('seconds must be within .1.. .5')
    args.out.mkdir(parents=True,exist_ok=True)
    result_path=args.out/'results.json'
    if result_path.exists(): raise FileExistsError(result_path)
    os.environ.setdefault('FLYGYM_ASSET_CACHE_DIR','/tmp/novigrad-flygym-assets-20260918')
    report={'scope':'Installation and headless CPU physics/controller-step feasibility only; not LM bridge, motor learning, robotics transfer or biological alignment validation.',
        'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'script_sha256':sha(__file__),'requested_seconds':args.seconds,'seed':0,
        'renderer':False,'backend':'MuJoCo CPU; no Warp','sources':[
        'https://neuromechfly.org/installation/','https://neuromechfly.org/tutorials/4c_hybrid_controller/',
        'https://github.com/NeLy-EPFL/flygym/tree/v2.1.0'],
        'licenses':'FlyGym Apache-2.0, separate from Novigrad MIT. Bundled NeuroMechFly assets and preprogrammed steps originate in the pinned FlyGym package; no third-party binary/assets redistributed here.'}
    installed=sorted((d.metadata['Name'],d.version) for d in md.distributions())
    freeze=''.join(f'{name}=={version}\n' for name,version in installed)
    (args.out/'requirements-freeze.txt').write_text(freeze)
    report['dependencies']={name:version for name,version in installed}
    report['distribution_metadata_hashes']={}
    report['asset_hashes']={}
    for d in md.distributions():
        for f in d.files or []:
            if str(f).endswith(('/METADATA','/RECORD')):
                report['distribution_metadata_hashes'][str(f)]=sha(d.locate_file(f))
            if d.metadata['Name'].lower()=='flygym' and ('/assets/' in str(f) or '/data/' in str(f)) and not str(f).endswith('.pyc'):
                p=d.locate_file(f)
                if p.is_file(): report['asset_hashes'][str(f)]={'sha256':sha(p),'bytes':p.stat().st_size}
    start=time.perf_counter()
    try:
        if md.version('flygym')!='2.1.0':raise ValueError('reproducer pins flygym==2.1.0')
        import numpy as np
        from flygym import Simulation
        from flygym.anatomy import BodySegment, ContactBodiesPreset
        from flygym.compose import MixedTerrainWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import HybridController, HybridControllerObservation, LocomotionAction, PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly
        fly=make_locomotion_fly(name='cpu_smoke',add_adhesion=True,colorize=False)
        world=MixedTerrainWorld()
        world.add_fly(fly,[0,0,1.2],Rotation3D('quat',[1,0,0,0]),bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,add_ground_contact_sensors=False)
        sim=Simulation(world)
        steps=PreprogrammedSteps(); dofs=fly.get_actuated_jointdofs_order('position')
        controller=HybridController(timestep=sim.timestep,preprogrammed_steps=steps,output_dof_order=dofs)
        sim.reset(); controller.reset(seed=0)
        apply_locomotion_action(sim,fly.name,LocomotionAction(joint_angles=steps.default_pose_by_dof_order(dofs),adhesion_onoff=np.ones(6,dtype=bool)))
        sim.warmup()
        report['setup_and_warmup_seconds']=time.perf_counter()-start
        index=fly.get_bodysegs_order().index(BodySegment('c_thorax'))
        initial=np.asarray(sim.get_body_positions(fly.name)[index]).copy()
        n=round(args.seconds/sim.timestep); start=time.perf_counter()
        trajectory=[]
        for i in range(n):
            action=controller.step(HybridControllerObservation.from_sim(sim,fly.name))
            apply_locomotion_action(sim,fly.name,action); sim.step_with_profile()
            position=np.asarray(sim.get_body_positions(fly.name)[index])
            if not np.isfinite(position).all(): raise ValueError(f'nonfinite thorax position at step {i}')
            if i%100==0 or i==n-1: trajectory.append({'step':i+1,'thorax_position_mm':position.tolist()})
        elapsed=time.perf_counter()-start
        report.update(status='passed',steps=n,timestep=float(sim.timestep),simulated_seconds=n*sim.timestep,
            step_wall_seconds=elapsed,steps_per_wall_second=n/elapsed,real_time_factor=n*sim.timestep/elapsed,
            actuated_dofs=len(dofs),initial_thorax_position_mm=initial.tolist(),trajectory=trajectory,
            final_displacement_mm=(position-initial).tolist())
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    cache=Path(os.environ['FLYGYM_ASSET_CACHE_DIR'])
    report['external_asset_cache']={str(p.relative_to(cache)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in cache.rglob('*') if p.is_file()} if cache.exists() else {}
    report['external_asset_endpoint']='https://datasets.epfl.ch/nely-public-share/flygym_assets/ (only used if lazy assets requested)'
    report['requirements_sha256']=sha(args.out/'requirements-freeze.txt')
    result_path.write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in (result_path,args.out/'requirements-freeze.txt')},'script':{'path':'examples/bio_bridge/flygym_smoke.py','sha256':sha(__file__)}},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['status','error','steps','step_wall_seconds','real_time_factor','final_displacement_mm']}))
    if report['status']!='passed': raise SystemExit(1)

if __name__=='__main__':main()
