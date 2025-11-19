import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import re

# Importar TensorFlow con manejo de errores para diferentes versiones
TENSORFLOW_AVAILABLE = False
try:
    # Intentar con la estructura tradicional de tensorflow.keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        # Intentar con la nueva estructura (TF 2.16+) donde keras es independiente
        from keras.preprocessing.text import Tokenizer
        from keras.preprocessing.sequence import pad_sequences
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        print("Advertencia: No se pudieron importar funciones de tokenización de Keras. La funcionalidad de tokenización estará limitada.")
        # Importar alternativas si TensorFlow no está disponible
        from typing import Optional
        # Definir valores por defecto
        Tokenizer = None
        pad_sequences = None


def preprocesar_texto_ingredientes(df: pd.DataFrame, columna_ingrediente: str = 'lista_ingredientes') -> pd.DataFrame:
    """
    Preprocesa la columna de ingredientes, convirtiendo la lista de ingredientes en texto
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna_ingrediente (str): Nombre de la columna que contiene los ingredientes
        
    Returns:
        pd.DataFrame: DataFrame con ingredientes procesados
    """
    df = df.copy()
    
    # Convertir la lista de ingredientes en una sola cadena de texto
    df['texto_ingredientes'] = df[columna_ingrediente].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    
    return df


def crear_etiquetas_nutricionales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea categorías nutricionales basadas en los valores numéricos
    
    Args:
        df (pd.DataFrame): DataFrame con datos nutricionales
        
    Returns:
        pd.DataFrame: DataFrame con nuevas columnas de categorías nutricionales
    """
    df = df.copy()
    
    # Categorizar según contenido de proteínas
    df['categoria_proteina'] = df['proteina'].apply(lambda x: 'alto' if x >= df['proteina'].quantile(0.67) else ('medio' if x >= df['proteina'].quantile(0.33) else 'bajo'))
    
    # Categorizar según contenido de carbohidratos
    df['categoria_carbohidratos'] = df['carbohidratos'].apply(lambda x: 'alto' if x >= df['carbohidratos'].quantile(0.67) else ('medio' if x >= df['carbohidratos'].quantile(0.33) else 'bajo'))
    
    # Categorizar según contenido de grasas
    df['categoria_grasa'] = df['grasa'].apply(lambda x: 'alto' if x >= df['grasa'].quantile(0.67) else ('medio' if x >= df['grasa'].quantile(0.33) else 'bajo'))
    
    # Categorizar por calorías
    df['categoria_calorias'] = df['calorias'].apply(lambda x: 'alto' if x >= df['calorias'].quantile(0.67) else ('medio' if x >= df['calorias'].quantile(0.33) else 'bajo'))
    
    return df


def tokenizar_ingredientes(df: pd.DataFrame, columna_texto: str = 'texto_ingredientes', max_vocab_size: int = 10000) -> Tuple:
    """
    Tokeniza la columna de texto de ingredientes

    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna_texto (str): Nombre de la columna con texto de ingredientes
        max_vocab_size (int): Tamaño máximo del vocabulario

    Returns:
        Tuple: Tokenizer entrenado y secuencias tokenizadas, o valores simulados si TensorFlow no está disponible
    """
    if TENSORFLOW_AVAILABLE:
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
        texts = df[columna_texto].values

        tokenizer.fit_on_texts(texts)

        sequences = tokenizer.texts_to_sequences(texts)

        # Hacer padding para que todas las secuencias tengan la misma longitud
        max_length = max([len(seq) for seq in sequences]) if sequences else 10
        sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

        return tokenizer, sequences
    else:
        # Implementación alternativa si TensorFlow no está disponible
        print("Usando implementación simulada de tokenización")
        texts = df[columna_texto].values
        # Simular tokenización simple basada en palabras
        word_to_idx = {'<PAD>': 0, '<OOV>': 1}
        all_sequences = []

        for text in texts:
            words = str(text).lower().split()
            sequence = []
            for word in words:
                if word not in word_to_idx and len(word_to_idx) < max_vocab_size:
                    word_to_idx[word] = len(word_to_idx)
                idx = word_to_idx.get(word, word_to_idx['<OOV>'])
                sequence.append(idx)
            all_sequences.append(sequence)

        # Pad secuencias manualmente
        if all_sequences:
            max_length = max(len(seq) for seq in all_sequences)
            padded_sequences = []
            for seq in all_sequences:
                padded = seq + [0] * (max_length - len(seq))  # Pad con 0
                padded_sequences.append(padded)
        else:
            padded_sequences = []

        # Devolver un objeto simulado con métodos básicos
        class MockTokenizer:
            def __init__(self, word_index):
                self.word_index = word_index

        mock_tokenizer = MockTokenizer(word_to_idx)
        return mock_tokenizer, np.array(padded_sequences)


def escalar_datos_nutricionales(df: pd.DataFrame, columnas: list = ['calorias', 'proteina', 'carbohidratos', 'grasa']) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Escala las columnas nutricionales usando StandardScaler
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columnas (list): Lista de columnas a escalar
        
    Returns:
        Tuple[pd.DataFrame, StandardScaler]: DataFrame escalado y el scaler ajustado
    """
    df = df.copy()
    scaler = StandardScaler()
    
    df[columnas] = scaler.fit_transform(df[columnas])
    
    return df, scaler


def codificar_etiquetas(df: pd.DataFrame, columna_etiqueta: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Codifica las etiquetas categóricas en valores numéricos
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        columna_etiqueta (str): Nombre de la columna con etiquetas
        
    Returns:
        Tuple[pd.DataFrame, LabelEncoder]: DataFrame con etiquetas codificadas y el encoder ajustado
    """
    df = df.copy()
    label_encoder = LabelEncoder()
    
    df[f'{columna_etiqueta}_codificado'] = label_encoder.fit_transform(df[columna_etiqueta])
    
    return df, label_encoder