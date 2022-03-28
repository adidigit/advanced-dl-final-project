import tarfile
import requests

main_path = 'C:/Users/naama-alon/data/'
paths = [['https://zenodo.org/record/2535967#.YjXGdHpBxaQ//CIFAR-10-C.tar', main_path + 'CIFAR-10-C.tar'],
        ['https://zenodo.org/record/3555552#.YjXGdXpBxaQ//CIFAR-100-C.tar', main_path + 'CIFAR-100-C.tar']]
#target_path = '~data/CIFAR-10-C.tar'
#'C:/Users/naama-alon/data/CIFAR-10-C.tar'

for url,target_path in paths:
    # cant download - need to fix
    #response = requests.get(url, stream=True)
    #if response.status_code == 200:
    #    with open(target_path, 'wb') as f:
    #        f.write(response.content)#raw.read())

    tar = tarfile.open(target_path)
    tar.extractall(main_path)
    tar.close()

