import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class BreastCancerModel {
    public static void main(String[] args) throws IOException, InterruptedException {
        double learingRate = 0.001;
        int numInputs = 9;
        int numHidden = 10;
        int numOutputs = 2;

        System.out.println("1 > Création du modèle");
        MultiLayerConfiguration configuration =
                new NeuralNetConfiguration.Builder()
                        .seed(1234)
                        .updater(new Adam(learingRate))
                        .weightInit(WeightInit.XAVIER)
                        .list()
                        .layer(0, new DenseLayer.Builder()
                                .nIn(numInputs)
                                .nOut(numHidden)
                                .activation(Activation.SIGMOID)
                                .build())
                        .layer(1, new OutputLayer.Builder()
                                .nIn(numHidden)
                                .nOut(numOutputs)
                                .activation(Activation.SOFTMAX)
                                .lossFunction(
                                        LossFunctions
                                                .LossFunction
                                                .MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .build())
                        .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        System.out.println("2 > Entrainement du modèle");

        File fileTrain = new ClassPathResource("breast-cancer-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain));
        int batchSize = 10;
        int classIndex = 9;
        DataSetIterator dataSetIteratorTrain =
                new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex, numOutputs);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        int numEpoches = 10;
        for (int i = 0; i < numEpoches; i++) {
            model.fit(dataSetIteratorTrain);
        }
        System.out.println("3 > Evaluation du modèle");

        File fileTest = new ClassPathResource("breast-cancer-test.csv").getFile();
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));

        DataSetIterator dataSetIteratorTest =
                new RecordReaderDataSetIterator(recordReaderTest, batchSize, classIndex, numOutputs);
        Evaluation evaluation = new Evaluation(numOutputs);

        while (dataSetIteratorTest.hasNext()) {
            DataSet dataSetTest = dataSetIteratorTest.next();
            INDArray features = dataSetTest.getFeatures();
            INDArray targetLables = dataSetTest.getLabels();
            INDArray predictedLables = model.output(features);
            evaluation.eval(predictedLables, targetLables);
        }

        System.out.println(evaluation.stats());

        ModelSerializer.writeModel(model, "BREAST-CANCER-Model.zip", true);

    }
}
