
def add_arguments(parser):

  # Environment options.
  parser.add_argument('--workers', default=3, type=int, metavar='<int>', help='Number of workers. At least 3 workers are needed. By default, 3. 1 trainer, 1 data preprocessor, 1 or more data generator.')
  parser.add_argument('--workdir', default='workdir', type=str, metavar='<folder>', help='Folder to same results. By default, "workdir"')
  parser.add_argument('--del-existing-workdir', action='store_true', help='Remove the workdir folder if exists.')
  parser.add_argument('--cuda', action='store_true', help='Use CUDA if available.')

