from setuptools import setup
import os
from glob import glob

package_name = 'cen449_project1'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # ament index kaydı
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # Tüm launch *.py dosyalarını yükle
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ajdar Civan Kırmızıgül',
    maintainer_email='acivankg@gmail.com',
    description='CEN449 Project 1 - TurtleBot3 navigation without pre-saved map',
    license='Proprietary',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goal_pose_client = cen449_project1.smart_goal_pose_vf:main',
        ],
    },
)

