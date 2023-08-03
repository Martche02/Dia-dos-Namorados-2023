import numpy as np
from PIL import Image
import os
import librosa
from pydub import AudioSegment
import eyed3

def npy2png(espectrograma, fase):
    espectrograma_normalizado = (espectrograma - np.min(espectrograma)) / (np.max(espectrograma) - np.min(espectrograma))
    fase_normalizada = (fase + np.pi) / (2 * np.pi)
    imagem = np.stack([espectrograma_normalizado]*3, axis=-1)
    imagem[..., 2] = fase_normalizada
    imagem = (imagem * 255).astype(np.uint8)
    imagem = Image.fromarray(imagem, mode='RGB')
    return imagem

def mp32npy(musica):
    audio, sr = librosa.load(musica, sr=None)
    # audio = librosa.effects.time_stretch(audio, rate=10)
    stft_resultado = librosa.stft(audio)
    espectrograma = np.abs(stft_resultado)
    fase = np.angle(stft_resultado)
    return espectrograma, fase

def png2npy(imagem):
    array_imagem = np.array(imagem) / 255.0
    espectrograma_normalizado = array_imagem[..., 0]
    fase_normalizada = array_imagem[..., 2]
    fase = fase_normalizada * 2 * np.pi - np.pi
    return espectrograma_normalizado, fase

def loadPng(arquivo):
    imagem = Image.open(arquivo)
    return imagem.convert('RGB')

def resizePng(imagem1, imagem2):
    tamanho_desejado = (imagem1.width, imagem1.height)
    imagem1_redimensionada = imagem1.resize(tamanho_desejado)
    imagem2_redimensionada = imagem2.resize(tamanho_desejado)
    return imagem1_redimensionada, imagem2_redimensionada

def fechadura(foto:str, espectrograma:str, local_de_entrega="musica.png"):
    '''png, mp3, png'''
    foto = loadPng(foto)
    espectrograma, fase = mp32npy(espectrograma)
    espectrograma = npy2png(espectrograma, fase)
    espectrograma, foto = resizePng(espectrograma, foto)
    foto = Image.fromarray((np.array(espectrograma) + np.array(foto)).astype(np.uint8))
    # imagem_resultante = imagem_resultante.resize((imagem_resultante.size[0], imagem_resultante.size[1]*10))
    foto.save(local_de_entrega)
    return local_de_entrega

def destrancar(chave:str, fechadura:str, local_de_entrega="musicadestravada.mp3"):
    '''png, png, mp3'''
    c=chave
    chave = loadPng(chave)
    fechadura = loadPng(fechadura)
    # fechadura = fechadura.resize((fechadura.size[0], fechadura.size[1]//4))
    fechadura, chave = resizePng(fechadura, chave)
    fechadura = Image.fromarray((np.array(fechadura) - np.array(chave)).astype(np.uint8))
    e, fase = png2npy(fechadura)
    e = e*np.max(np.abs(e))
    e = e*np.exp(1j * fase)
    e = librosa.istft(e)
    e = e/np.max(e)
    e = (e * np.iinfo(np.int16).max).astype(np.int16)
    e = e.tobytes()
    audio_segment = AudioSegment(
        data=e,
        sample_width=2,  # 2 bytes = 16 bits
        frame_rate=44100,
        channels=1)
    audio_segment.export(local_de_entrega, format="mp3")
    metaMusica = eyed3.load(local_de_entrega)
    if metaMusica.tag is None:
        metaMusica.initTag()
    metaMusica.tag.images.set(3, open(c,'rb').read(), 'image/png')
    metaMusica.tag.save()
    return os.path.join(os.getcwd(), local_de_entrega)