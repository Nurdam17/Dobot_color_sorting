from setuptools import setup

package_name = 'dobot_sorting_color'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aliyanur',
    maintainer_email='aliyanur@todo.todo',
    description='Color sorting with Dobot in ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_new = dobot_sorting_color.camera_new:main',
            'main_new = dobot_sorting_color.main_new:main',
        ],
    },
)