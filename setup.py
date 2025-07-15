from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DASH = '-e .'

def get_requirements(file_path:str) -> List[str]:
    """This function will return a list of requirement"""
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n', '') for req in requirements]

        if HYPHEN_E_DASH in requirements:
            requirements.remove(HYPHEN_E_DASH)

    pass


setup(
    name="mlproject" ,
    version="0.1",
    author='Suraj Aryan',
    author_email='surajaryancuj14@gmal.com',
    description='A machine learning project for predicting outcomes',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)