"""This test module runs standard tests to confirm modifications to the software"""
import os
import subprocess

if __name__ == '__main__':
    # Clone regression tests repository in this directory
    print('Regression test started.')
    if not os.path.exists('../../Regression'):
        print('Downloading repository...')
        subprocess.run(['../../download_benchmarks.sh', 'Regression'])

    print('Benchmarks repository found.')
    for dirname in os.listdir('Regression'):
        if os.path.isdir(os.path.join('Regression', dirname)):
            print('Running benchmarks for {}...'.format(dirname))
            subprocess.run(
                ['../../never2_batch.py',
                 '-o', f'results_batch_{dirname}',
                 f'Regression/{dirname}/instances.csv',
                 'ssbp'])
            print('Completed.')

    print('regression test completed. Results are saved in CSV.')
