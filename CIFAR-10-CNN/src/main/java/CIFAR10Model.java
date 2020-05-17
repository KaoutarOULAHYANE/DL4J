import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CIFAR10Model {

    private static int height = 32;
    private static int width = 32;
    private static int channels = 3;
    private static int output = 10;
    private static int batchSize = 64;
    private static long seed = 123L;
    private static int epochs = 10;
    private static double learningRate = 0.001;

    public static void main(String[] args) throws IOException {

        System.out.println("> Création du modèle");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nIn(channels)
                        .nOut(32)
                        .build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer
                        .Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(
                                SubsamplingLayer.PoolingType.MAX)
                        .build())

                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(16)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(64)
                        .build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer
                        .Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(
                                SubsamplingLayer.PoolingType.MAX)
                        .build())

                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(32)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(128)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(64)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer
                        .Builder()
                        .kernelSize(1, 1)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.ELU)
                        .nOut(output)
                        .build())
                .layer(new BatchNormalization())

                .layer(new SubsamplingLayer
                        .Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())

                .layer(new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(output)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(
                        InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println("> Entrainement du modèle");
        String path = System.getProperty("user.home") + "/CIFAR-10/";
        File fileTrain = new File(path + "/train");
        FileSplit fileSplitTrain =
                new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, new Random(1234));
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
        ImageRecordReader imageRecordReaderTrain = new ImageRecordReader(
                height, width, channels, labelGenerator
        );
        imageRecordReaderTrain.initialize(fileSplitTrain);

        int labeIndex = 1;

        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(
                imageRecordReaderTrain, batchSize, labeIndex, output);

        DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);
        dataSetIteratorTrain.setPreProcessor(dataNormalization);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        for (int i = 0; i < epochs; i++) {
            System.out.println("Etape " + (i + 1));
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("> Evaluation du modèle");

        File fileTest = new File(path + "/test");
        FileSplit fileSplitTest =
                new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS, new Random(1234));
        ImageRecordReader imageRecordReaderTest = new ImageRecordReader(
                height, width, channels, labelGenerator
        );
        imageRecordReaderTest.initialize(fileSplitTest);

        DataSetIterator dataSetIteratorTest =
                new RecordReaderDataSetIterator(imageRecordReaderTest, batchSize, labeIndex, output);
        Evaluation evaluation = new Evaluation(output);

        int step = 1;
        while (dataSetIteratorTest.hasNext()) {
            DataSet dataSetTest = dataSetIteratorTest.next();
            INDArray features = dataSetTest.getFeatures();
            INDArray targetLables = dataSetTest.getLabels();
            INDArray predictedLables = model.output(features);
            evaluation.eval(predictedLables, targetLables);
            System.out.println("Etape " + step++);
        }

        System.out.println(evaluation.stats());

        ModelSerializer.writeModel(model, "CIFAR-10-Model.zip", true);
    }
}