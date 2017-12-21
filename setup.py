from setuptools import setup, find_packages

setup(
    name='shade',
    version='0.3',
    packages=find_packages(),
    zip_safe=False,
    package_data={
        'shade': ['data/*.fits'],
    },
    include_package_data=True,
    #install_requires=['Pillow'],
)
