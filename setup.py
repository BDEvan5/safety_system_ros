from setuptools import setup
import os
from glob import glob

package_name = 'safety_system_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('map_data/*.csv')),
        (os.path.join('share', package_name), glob('safety_system_ros/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benjy',
    maintainer_email='benjaminevans316@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_node=safety_system_ros.SimpleNode:main',
            'pure_pursuit=safety_system_ros.PurePursuit:main',
        ],
    },
)
