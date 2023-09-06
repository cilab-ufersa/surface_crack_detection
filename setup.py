import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='surface_crack_detection',
    url='https://github.com/cilab-ufersa/surface_crack_detection',
    author='CILAB',
    author_email='cilab.ufersa@gmail.com',
    # Needed to actually package something
    packages=setuptools.find_packages(),
    include_package_data=True,
    # Needed for dependencies
    install_requires=required,
    description='A package to detect surface crack',
    long_description=open('README.md').read(),
)