#!/usr/bin/env python3
"""Three independently trained connectome policies, fresh held-out driving votes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

from train_driving_game import ROOT, train, server, evaluate, episode, EPISODES
from ensemble_policy import EnsembleClient
from driving_environment import CONFIG, ACTION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/magi')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    validation = list(range(30000, 30008))
    test = list(range(40000, 40030))
    seeds = [(11, 5000), (22, 6000), (33, 7000)]
    protocol = {'members': 3, 'member_rng_and_environment_starts': seeds,
                'episodes_per_member': EPISODES, 'learning_rate': .01,
                'validation_seeds': validation, 'test_seeds': test,
                'environment': CONFIG, 'actions': ACTION_NAMES,
                'voting': ['majority with mean-probability tie break', 'equal probability mean'],
                'reset': 'identical initial topology and weights; independent exploration and training environment seeds',
                'selection': 'each checkpoint and best single selected on validation before test; no ensemble tuning'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    members, scores = [], []
    for index, (rng, start) in enumerate(seeds):
        folder = out/f'member{index+1}'
        folder.mkdir()
        initial = folder/'initial.safetensors'
        subprocess.run([str(ROOT/'target/release/novi_engine'), 'init', str(ROOT/'data/pn_kc.tsv'),
                        str(ROOT/'data/kc_mbon.tsv'), str(initial), '5', '.01', '6', '.2', 'false', 'opponent'],
                       check=True, capture_output=True)
        result = train(initial, folder, rng_seed=rng, training_seed_start=start, validation_seeds=validation)
        members.append(folder/'best.safetensors')
        scores.append(result['best_validation_return'])
    best = max(range(3), key=lambda i: scores[i])
    selection = {'validation_returns': scores, 'best_single_member': best+1,
                 'member_sha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in members]}
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    results = {}
    with ExitStack() as stack:
        clients = [stack.enter_context(server(path, out/f'eval-member{i+1}')) for i, path in enumerate(members)]
        for i, client in enumerate(clients):
            results[f'member{i+1}'] = evaluate(client, test)
        for mode in ('majority', 'mean'):
            ensemble = EnsembleClient(clients, mode)
            result = evaluate(ensemble, test)
            result['voting_statistics'] = ensemble.statistics()
            (out/f'{mode}-decisions.json').write_text(json.dumps(ensemble.rows, indent=2)+'\n')
            results[mode] = result
        results['always_slow'] = evaluate(clients[0], test, 'slow')
        # Render a predetermined first test seed; no outcome-based scene selection.
        ensemble = EnsembleClient(clients, 'majority')
        _, _, _, frames = episode(ensemble, test[0], render=True)
        if frames:
            frames[0].save(out/'voting.gif', save_all=True, append_images=frames[1:], duration=500, loop=0)
    results['best_validation_single'] = {'member': best+1, **results[f'member{best+1}']}
    summary = {'results': results, 'platform': platform.platform(), 'training_episodes_total': 3*EPISODES,
               'note': 'three full models vs one; disagreement is measured on common observations within each ensemble trajectory, not compared across different trajectories',
               'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), ROOT/'scripts/train_driving_game.py', ROOT/'scripts/ensemble_policy.py']}}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'bundle.json').write_text(json.dumps({'format': 'novi.ensemble.v1', 'members': [str(p.relative_to(out)) for p in members],
                                              'default_mode': 'majority', 'actions': ACTION_NAMES,
                                              'encoding': 'driving_environment.encode_observation'}, indent=2)+'\n')
    print(json.dumps({name: result['aggregate'] for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
