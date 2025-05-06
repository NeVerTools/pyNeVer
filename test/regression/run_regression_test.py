"""This test module runs standard tests to confirm modifications to the software"""
import os
import subprocess

if __name__ == '__main__':
    # Clone regression tests repository in this directory
    print('Regression test started.')
    if not os.path.exists('Regression'):
        print('Downloading repository...')
        subprocess.run(['./download_benchmarks.sh', 'Regression'])

    print('Benchmarks repository found.')
    for dirname in ['ACAS_XU', 'RL']:  # CIFAR and resnet to come
        print('Running benchmarks for {}...'.format(dirname))
        subprocess.run(
            ['python', 'never2_batch.py',
             '-o', f'results_batch_{dirname}.csv',
             f'Regression/{dirname}/instances.csv',
             'ssbp'])
        print('Completed.')

        # Check all verified
        with open(f'results_batch_{dirname}.csv', 'r') as f:
            for line in f:
                if line.strip('\n').split(',')[2] != 'Verified':
                    print('Test failed!')
                    exit(1)

    print('Regression test completed. All tests passed.')
    exit(0)
