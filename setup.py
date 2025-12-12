from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'pointcloud_detection_3d'

setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # Install scripts to lib folder
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Andrés Islas Bravo',
    maintainer_email='andresislas99@icloud.com',
    description='Real-time 3D Object Detection for LiDAR Point Clouds',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node.py = scripts.detection_node:main',
        ],
    },
    scripts=[
        'scripts/detection_node.py',
        'scripts/benchmark.py',
        'scripts/pointpillars_inference.py',
    ],
    python_requires='>=3.8',
)
