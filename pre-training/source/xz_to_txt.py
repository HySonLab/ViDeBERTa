import lzma
import argparse
import lzma
from pyvi import ViTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file', type=str, default=None, help='Source file')
    parser.add_argument('--destination_file', type=str, default=None, help='Destination dir')
    # parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cnt = 0
    # lzma._BUFFER_SIZE = 1023
    with open(args.destination_file, mode = "wt") as fout:
        with lzma.open(args.source_file, mode = "rt") as fin:
            for line in fin:
                cnt+=1
                if len(line) > 0:
                    try: 
                        fout.write(ViTokenizer.tokenize(line))
                        fout.write(" \n")
                    except:
                        print(cnt)
                        print(line)
                if(cnt%1e6 == 0):
                    print(cnt)
                    print(line)
                if(cnt%2e8 == 0):
                    break
