import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("models/heart_cnn_final.h5")

print("Saving optimized model...")
model.save("models/heart_cnn_final_optimized.h5", include_optimizer=False)

print("✅ Optimization complete!")
