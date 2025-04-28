from setuptools import setup, find_packages

setup(
    name='traffic-signs-classifier',
    version='0.1.0',
    author='Ayush Poojari',
    author_email='poojari.ayush@ucdconnect.ie',
    description='A deep learning project for classifying traffic signs using deep learning models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AyushPoojariUCD/traffic-signs-classifier',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tqdm==4.64.0',
        'gdown==4.4.0',
        'python-dotenv==1.0.0',
        'pyyaml==6.0',
        'ruamel.yaml==0.17.21',
        'opencv-python',
        'numpy==1.21.0',
        'pandas==1.3.3',
        'seaborn==0.11.0',
        'matplotlib==3.4.3',
        'tensorflow==2.17.0',
        'scikit-learn==1.0.0',
        'jupyter==1.0.0',
        'ipykernel==6.0.0',
        'jupyterlab==3.4.0'
    ],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Computer Science :: Deep Learning'
    ],
    entry_points={
        'console_scripts': [
            'train-model = pipeline.train:main',
            'evaluate-model = pipeline.evaluate:main',
        ],
    }
)
