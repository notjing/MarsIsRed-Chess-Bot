import tensorflow as tf
import tf2onnx

def convert_model():
    model_path = "model_iteration/chessai_model_v7.1.keras"
    onnx_path = "../model_cache/model_v7.1.onnx"

    print("Loading Keras model...")
    model = tf.keras.models.load_model(
        model_path,
    )

    print("Converting to ONNX...")
    # Define the exact input shapes and data types the model expects
    # 'None' allows for dynamic batch sizes during search
    input_signature = [
        tf.TensorSpec((None, 8, 8, 25), tf.float32, name="board_input"),
        tf.TensorSpec((None, 19), tf.float32, name="extra_input")
    ]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_path
    )
    print(f"Success! Optimized model saved to {onnx_path}")

if __name__ == "__main__":
    convert_model()
