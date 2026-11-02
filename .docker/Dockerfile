# 1. Partiamo dall'ambiente base con Ubuntu e ROS 2 Humble già pronti e isolati
FROM osrf/ros:humble-desktop-full

# 2. Installiamo il gestore pacchetti pip per il Python INTERNO a Docker
RUN apt-get update && apt-get install -y python3-pip git

# 3. Installiamo le dipendenze Python a livello globale DENTRO Docker.
# (Non serve Conda/venv perché l'intero container è già un ambiente isolato)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install numpy scipy

# 4. Impostiamo la cartella di lavoro
WORKDIR /workspace

# 5. Diciamo al container cosa fare quando si avvia (aprire il terminale bash)
CMD ["/bin/bash"]



