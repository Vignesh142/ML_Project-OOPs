from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str)->List[str]:
    '''
    Read the requirements file and return the list of requirements
    '''
    requirements = []
    with open(file_path, 'r') as f:
        for line in f:
            requirements.append(line.strip())
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Vignesh',
    author_email="ajaypunna9342@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
)

# if __name__ == '__main__':
#     print(get_requirements('requirements.txt'))
    