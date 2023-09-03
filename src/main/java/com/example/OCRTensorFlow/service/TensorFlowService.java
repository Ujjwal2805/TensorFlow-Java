package com.example.OCRTensorFlow.service;
import org.springframework.stereotype.Service;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

@Service
public class TensorFlowService {

    private final String modelPath = "classpath:/mobilenet_v2_130_224/classification/4";
    private final List<String> labels;

    public TensorFlowService() throws IOException {
        // Load labels for image classes
        this.labels = Files.readAllLines(Path.of("classpath:/mobilenet_v2_130_224/classification/4/labels.txt"));
    }

    public String classifyImage(byte[] imageData) {
        try (Graph graph = new Graph()) {
            byte[] graphBytes = Files.readAllBytes(Path.of(modelPath));
            graph.importGraphDef(graphBytes);

            try (Session session = new Session(graph)) {
                Tensor<String> imageTensor = Tensors.create(imageData);
                List<Tensor<?>> result = session.runner()
                        .feed("input_1", imageTensor)
                        .fetch("Identity")
                        .run();

                Tensor<?> outputTensor = result.get(0);
                long[] labelIndices = outputTensor.copyTo(new long[1]);
                int predictedClass = (int) labelIndices[0];

                return labels.get(predictedClass);
            }
        } catch (IOException e) {
            e.printStackTrace();
            return "Error classifying image.";
        }
    }
}


