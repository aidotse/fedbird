import tempfile
from src.client1_new import TrainingProcess, Model, TrainDataReader

def create_seed_model():
    model = Model()
    data = TrainDataReader()
    start_process = TrainingProcess(data, model,
                                    classes_path='../data/model_data/seabird_classes.txt',
                                    anchors_path='../data/model_data/tiny_yolo_anchors.txt',
                                    data_path='../data/Annotation/list1.txt')


    return start_process.local_model


def save_model(outer_model, path='package'):
    import tarfile

    _, weights_path = tempfile.mkstemp(suffix='.h5')
    outer_model['model'].save_weights(weights_path)
    tar = tarfile.open(path, "w:gz")
    tar.add(weights_path,'weights.h5')
    tar.add('src', 'src')
    tar.close()

    return path


if __name__ == '__main__':
    outer_model = {}
    outer_model['model'] = create_seed_model()
    outfile_name = "birdcagenew"
    save_model(outer_model, outfile_name)
    print("seed model saved as: ", outfile_name)
