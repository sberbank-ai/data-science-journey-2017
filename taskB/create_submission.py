import argparse
import zipfile
import os
import ntpath
import tqdm


def get_files_and_check_it(files):
    result_files = []
    for filename in files:
        if not os.path.exists(filename):
            raise Exception("file '{}', which you want to put in submission, dont exist".format(filename))
        if os.path.isdir(filename):
            folder = filename
            for filename in os.listdir(folder):
                new_filename = os.path.join(folder, filename)
                if os.path.isdir(new_filename):
                    for filename in get_files_and_check_it([new_filename]):
                        result_files.append(filename)
                else:
                    result_files.append(new_filename)
        else:
            result_files.append(filename)
    return result_files


def create_archive(output, files):
    """
    put all files (include directories) into output zip file
    remove all pathes
    """
    result_files = get_files_and_check_it(files)

    if len(result_files) == 0:
        raise Exception("empty file list")

    if os.path.exists(output):
        raise Exception("file '{}' already exists".format(output))

    metadata_jsons = [filename for filename in result_files if ntpath.basename(filename) == "metadata.json"]
    if len(metadata_jsons) != 1:
        raise Exception("should exists one metadata.json, found: {}".format(str(len(metadata_jsons))))

    with zipfile.ZipFile(output, 'w') as result_zip:
        for filename in tqdm.tqdm(result_files):
            result_zip.write(filename, ntpath.basename(filename))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission')
    parser.add_argument('-p', '--path', action='append', help='files for zip archive, if path is directory put all files in directory without tree structure', required=True)
    parser.add_argument('-o', '--output', action='store', help='filename of zip archive', default="output.zip")
    args = parser.parse_args()
    print(args)
    exit(create_archive(args.output, args.path))
