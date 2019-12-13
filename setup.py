from setuptools import setup

descr = """sknw: skeleton analysis in Python.
Inspired by Juan Nunez-Iglesias's skan.
"""

if __name__ == '__main__':
    setup(name='sknw',
        version='0.13',
        url='https://github.com/yxdragon/sknw',
        description='Analysis of object skeletons',
        long_description=descr,
        author='YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=['sknw'],
        package_data={},
        install_requires=[
            'numpy',
            'networkx',
            'numba'
        ],
    )
