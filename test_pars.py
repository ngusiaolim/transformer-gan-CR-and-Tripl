import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_true',
                    help='show')

opt = parser.parse_args()
print(opt.show)