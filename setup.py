from setuptools import find_packages, setup

setup(
    name='transnormer-v2-pytorch',
    version='0.0.0',
    packages=find_packages(),
    description='Transnormer v2',
    author='Doraemonzzz',
    author_email='doraemon_zzz@163.com',
    url='https://github.com/Doraemonzzz/transnormer-v2-pytorch',
    install_requires=[
        'torch',
        'einops',
    ],
    keywords = [
        'artificial intelligence',
        'sequential model',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

)
