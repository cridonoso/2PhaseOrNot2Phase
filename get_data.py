from absl import app
from absl import flags
from google_drive_downloader import GoogleDriveDownloader as gdd


dataset_id = {
	'raw':{
		'wise': '1O-cFXWjuTMNNfWjVQCyEAa2IoAJFSPI6',
		'ogle': '1whdP_2SdMSHcODu8ItTWfM0tI-frGhie',
		'macho': '1zFkrP2TCYoGa3yWkrtymVDmWjLlViIpp',
		'linear': '1AWe-W7GX0jDGZ-NW6Nmz77J-LG5XP1c3',
		'gaia': '1VgG2RNi88VcVnpE_MgOYPFjS8A42KXTJ',
		'css': '1ZgSMYetSss8hqLMEBXhVjgo2hCHFdMBf',
		'asas': '1ouQmBedVok5LNH2u2aEQjAvl0KchrmTI',
	},
	'record':{
		'wise': '1hBrSxw71qaJW09MX7iZTrbUVU5vie4oC',
		'ogle': '17fIJ2L6jNjE1J1OkJ9jksGfXvH4CGzI7',
		'macho': '13en9ltKt4NacRw2ETTxAbCgc5EP3YrbK',
		'linear': '1iGuVxaWVTUpMihrn9IdP6CbQzxA5d7sZ',
		'gaia': '1Bg2clQGHwmiEOyH06KwAx17_0wDH4rr3',
		'css': '1PSRvc07tuR-zaQJbZGBboKcIitza2x18',
		'asas': '1e8ewHz0VOubbNyd3Y4D7uiqF6qW4-iAO',
	}
}

FLAGS = flags.FLAGS
flags.DEFINE_boolean('record', False, 'Get record if available')
flags.DEFINE_boolean('unzip', True, 'Unzip compressed file')

flags.DEFINE_string("destination", "./datasets/", "Folder for saving files")

flags.DEFINE_string("dataset", "macho", "Dataset to be downloaded (macho, linear, asas, wise, gaia, css, ogle)")


def main(argv):
	if FLAGS.record:
		file_id = dataset_id['record'][FLAGS.dataset]
		dest_path='{}/records/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)
	else:
		file_id = dataset_id['raw'][FLAGS.dataset]
		dest_path='{}/raw_data/{}/{}.zip'.format(FLAGS.destination, FLAGS.dataset, FLAGS.dataset)


	gdd.download_file_from_google_drive(file_id=file_id,
										dest_path=dest_path,
										unzip=FLAGS.unzip)


if __name__ == '__main__':
	app.run(main)

