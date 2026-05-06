from setuptools import setup
import os
from glob import glob

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='piyush',
    maintainer_email='piyush@todo.todo',
    description='YOLOv8 detector with custom messages',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = yolo_detector.detector_node:main',
        ],
    },
)

