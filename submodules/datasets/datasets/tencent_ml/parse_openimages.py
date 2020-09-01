import sys
import csv

def train():
    base_path = '/data/public/rw/datasets/tencent-ml'

    with open(base_path + '/metadata/downloaded_image_ids.csv', 'r') as f:
        reader = csv.DictReader(f)
        download_list = {row['ImageID']:0 for row in reader}
    print('processed downloaded_image_ids.csv', file=sys.stderr)

    with open(base_path + '/metadata/image_ids_and_rotation.csv', 'r') as f:
        reader = csv.DictReader(f)
        lines = [{key:value for key, value in row.items() if key in ['ImageID', 'Subset', 'OriginalURL']} for row in reader]
        lines = [line for line in lines if line['ImageID'] in download_list]
        reverse_pairs = {line['OriginalURL']:line['ImageID'] for line in lines}
    print('processed image_ids_and_rotation', file=sys.stderr)

    with open(base_path + '/metadata/train_urls_from_openimages.txt', 'r') as f:
        lines = [[l.strip() for l in line.strip().split('\t')] for line in f.readlines() if line.strip()]
        for line in lines:
            img_url, labels = line[0], [int(t.split(':')[0]) for t in line[1:]]
            if img_url not in reverse_pairs:
                print('not exist image url:' + img_url, file=sys.stderr)
                continue
            img_id = reverse_pairs[img_url]
            merged = [img_id, img_url] + [str(l) for l in labels]
            print('\t'.join(merged))
    print('processed train_urls_from_openimages.txt', file=sys.stderr)


def valid():
    base_path = '/data/public/rw/datasets/tencent-ml'

    with open(base_path + '/metadata/downloaded_image_ids.csv', 'r') as f:
        reader = csv.DictReader(f)
        download_list = {row['ImageID']:0 for row in reader}
    print('processed downloaded_image_ids.csv', file=sys.stderr)

    with open(base_path + '/metadata/validation-images-with-rotation.csv', 'r') as f:
        reader = csv.DictReader(f)
        lines = [{key:value for key, value in row.items() if key in ['ImageID', 'Subset', 'OriginalURL']} for row in reader]
        lines = [line for line in lines if line['ImageID'] in download_list]
        reverse_pairs = {line['OriginalURL']:line['ImageID'] for line in lines}
    print('processed validation-images-with-rotation.csv', file=sys.stderr)

    with open(base_path + '/metadata/val_urls_from_openimages.txt', 'r') as f:
        lines = [[l.strip() for l in line.strip().split('\t')] for line in f.readlines() if line.strip()]
        for line in lines:
            img_url, labels = line[0], [int(t.split(':')[0]) for t in line[1:]]
            img_id = reverse_pairs[img_url]
            merged = [img_id, img_url] + [str(l) for l in labels]
            print('\t'.join(merged))
    print('processed val_urls_from_openimages.txt', file=sys.stderr)


def imagenet_train():
    base_path = '/data/public/rw/datasets/tencent-ml'

    with open(base_path + '/metadata/imagenet_train_list.txt', 'r') as f:
        ids = {line.strip().split(' ')[0]:0 for line in f.readlines()}
    print('processed imagenet_train_list.txt', file=sys.stderr)

    with open(base_path + '/metadata/train_image_id_from_imagenet.txt', 'r') as f:
        for line in f.readlines():
            mid = line.split('\t')[0].split('.')[0]
            if mid in ids:
                print(line.strip())
            else:
                print('not exist image id:' + mid, file=sys.stderr)
    print('processed train_image_id_from_imagenet.txt', file=sys.stderr)


def imagenet_valid():
    base_path = '/data/public/rw/datasets/tencent-ml'

    with open(base_path + '/metadata/imagenet_train_list.txt', 'r') as f:
        ids = {line.strip().split(' ')[0]:0 for line in f.readlines()}
    print('processed imagenet_train_list.txt', file=sys.stderr)

    with open(base_path + '/metadata/val_image_id_from_imagenet.txt', 'r') as f:
        for line in f.readlines():
            mid = line.split('\t')[0].split('.')[0]
            if mid in ids:
                print(line.strip())
            else:
                print('not exist image id:' + mid, file=sys.stderr)
    print('processed val_image_id_from_imagenet.txt', file=sys.stderr)


if __name__ == '__main__':
    # train()
    # > python parse_openimages.py > metadata/train_ids_from_openimages.txt

    # valid()
    # > python parse_openimages.py > metadata/val_ids_from_openimages.txt

    # imagenet_train()
    # > python parse_openimages.py > metadata/train_image_id_from_imagenet_1k.txt

    imagenet_valid()
    # > python parse_openimages.py > metadata/val_image_id_from_imagenet_1k.txt
