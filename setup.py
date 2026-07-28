from setuptools import setup, find_packages

setup(
    name='agent-eval',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'pandas',
        'tabulate',
    ],
    entry_points={
        'console_scripts': [
            'agent-eval=agent_eval.cli:cli',
        ],
    },
    author='AutoScout',
    description='A prototype for agentic AI system observability and evaluation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/agent-eval-prototype', # Placeholder
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
