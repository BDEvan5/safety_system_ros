import glob , yaml, csv

import rosbag2_py  # noqa
import ros2bag.api.

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options



class Extraction:
    def __init__(self, eval_name=None):
        self.path = "Documents/USA Safety Data/"

        folders = glob.glob(f"{self.path}*")
        for i, folder in enumerate(folders):
            print(f"Folder being opened: {folder}")

            storage_options, converter_options = get_rosbag_options(bag_path)
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)

            topic_types = reader.get_all_topics_and_types()

            print(topic_types)

#  bag_path = str(RESOURCES_PATH / 'talker')
#     storage_options, converter_options = get_rosbag_options(bag_path)

#     reader = rosbag2_py.SequentialReader()
#     reader.open(storage_options, converter_options)

#     topic_types = reader.get_all_topics_and_types()

#     # Create a map for quicker lookup
#     type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}

#     # Set filter for topic of string type
#     storage_filter = rosbag2_py.StorageFilter(topics=['/topic'])
#     reader.set_filter(storage_filter)

#     msg_counter = 0

#     while reader.has_next():
#         (topic, data, t) = reader.read_next()
#         msg_type = get_message(type_map[topic])
#         msg = deserialize_message(data, msg_type)

#         assert isinstance(msg, String)
#         assert msg.data == f'Hello, world! {msg_counter}'

#         msg_counter += 1

#     # No filter
#     reader.reset_filter()

#     reader = rosbag2_py.SequentialReader()
#     reader.open(storage_options, converter_options)

#     msg_counter = 0

#     while reader.has_next():
#         (topic, data, t) = reader.read_next()
#         msg_type = get_message(type_map[topic])
#         msg = deserialize_message(data, msg_type)

#         assert isinstance(msg, Log) or isinstance(msg, String)

#         if isinstance(msg, String):
#             assert msg.data == f'Hello, world! {msg_counter}'
#             msg_counter += 1

Extraction()