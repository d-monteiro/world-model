# Guia Prático de Treino e Simulação: World Model (Fase 1)

**Autor:** Manus AI

**Data:** 2 de Fevereiro de 2026

## Introdução

Esta é a continuação do nosso guia de implementação. Aqui, vamos focar-nos no **como**: como treinar e simular o seu sistema de Fase 1. A escolha mais inteligente para a simulação, especialmente sem câmara, é o **MuJoCo (Multi-Joint dynamics with Contact)**.

**Porquê o MuJoCo?**

- **Velocidade e Precisão:** É um dos simuladores de física mais rápidos e precisos disponíveis, otimizado para dinâmicas de contacto, o que é essencial para manipulação robótica.
- **Padrão da Indústria:** É amplamente utilizado em investigação de robótica e reinforcement learning (DeepMind, OpenAI, etc.).
- **Excelentes Bindings Python:** Integra-se perfeitamente com o ecossistema Python, especialmente com a biblioteca `gymnasium`, que fornece uma interface padrão para ambientes de RL.

Vamos estruturar este guia em três partes: o setup do ambiente, o pipeline de treino detalhado com código, e como executar o agente final.

## 1. Setup do Ambiente de Simulação (MuJoCo)

O coração de uma simulação MuJoCo é um ficheiro XML que define o mundo: o robô, os objetos, as suas propriedades físicas e as suas relações.

### 1.1. O Ficheiro XML do Mundo (`robot_world.xml`)

Crie um ficheiro chamado `robot_world.xml`. Este ficheiro irá descrever um braço robótico simples de 3 juntas, um objeto cúbico que ele pode manipular, e um local-alvo.

```xml
<mujoco model="3_joint_arm">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>

  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="1 1 0.1" type="plane"/>
    
    <!-- Braço Robótico -->
    <body name="base" pos="0 0 0.1">
      <joint type="free"/>
      <geom name="base_geom" pos="0 0 0" size="0.05 0.05 0.05" type="box"/>
      <body name="link1" pos="0 0 0.05">
        <joint axis="0 0 1" name="joint1" pos="0 0 0" range="-180 180" type="hinge"/>
        <geom fromto="0 0 0 0.3 0 0" name="link1_geom" size="0.04" type="capsule"/>
        <body name="link2" pos="0.3 0 0">
          <joint axis="0 1 0" name="joint2" pos="0 0 0" range="-120 120" type="hinge"/>
          <geom fromto="0 0 0 0.3 0 0" name="link2_geom" size="0.04" type="capsule"/>
          <body name="link3" pos="0.3 0 0">
            <joint axis="0 1 0" name="joint3" pos="0 0 0" range="-120 120" type="hinge"/>
            <geom fromto="0 0 0 0.2 0 0" name="link3_geom" size="0.04" type="capsule"/>
            <site name="end_effector" pos="0.2 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Objeto a ser manipulado -->
    <body name="object" pos="0.5 0.1 0.05">
        <joint type="free" damping="0.01"/>
        <geom name="object_geom" size="0.05 0.05 0.05" type="box" rgba="0.2 0.2 1.0 1"/>
    </body>

    <!-- Alvo -->
    <site name="target" pos="-0.5 -0.5 0.1" size="0.05" rgba="1 0.2 0.2 0.5" type="sphere"/>

  </worldbody>

  <actuator>
      <motor joint="joint1" ctrllimited="true" ctrlrange="-1.0 1.0" gear="150.0"/>
      <motor joint="joint2" ctrllimited="true" ctrlrange="-1.0 1.0" gear="150.0"/>
      <motor joint="joint3" ctrllimited="true" ctrlrange="-1.0 1.0" gear="150.0"/>
  </actuator>
</mujoco>
```

### 1.2. Interagindo com o Ambiente em Python

Agora, vamos usar o `gymnasium` para carregar e controlar este mundo. O Gymnasium fornece uma interface `Env` padrão com métodos como `step()` e `reset()`.

**Instalação:**
```bash
sudo pip3 install gymnasium gymnasium[mujoco]
```

**Código Python (`env_test.py`):**

```python
import gymnasium as gym
import time

# Carregar o ambiente a partir do ficheiro XML
env = gym.make('Reacher-v4', xml_file='robot_world.xml') # Usamos um ambiente base como o Reacher para a estrutura

# Reiniciar o ambiente para obter a observação inicial
observation, info = env.reset()

for _ in range(1000):
    # Ação aleatória: um vetor com 3 valores entre -1 e 1 para os 3 motores
    action = env.action_space.sample()
    
    # Executar a ação no ambiente
    observation, reward, terminated, truncated, info = env.step(action)

    # O 'observation' contém o estado do mundo!
    # Para o Reacher-v4, isto inclui:
    # - cos(joint_angles), sin(joint_angles)
    # - joint_velocities
    # - end_effector_to_target_vector
    
    # Vamos extrair o nosso vetor de estado personalizado
    joint_angles = observation[:3] # Exemplo, pode precisar de ajuste
    object_pos = env.data.get_body_xpos('object')
    target_pos = env.data.get_site_xpos('target')
    
    state_vector = list(joint_angles) + list(object_pos) + list(target_pos)
    print(f"Vetor de Estado Personalizado: {len(state_vector)} dimensões")

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

Com isto, tem um ambiente de simulação funcional e uma forma de extrair o vetor de estado de baixa dimensão necessário para a Fase 1. O próximo passo é usar este ambiente para treinar os seus modelos.

## 2. Pipeline de Treino Detalhado (com Código PyTorch)

Agora que o ambiente está pronto, vamos ao treino de cada componente do World Model. Usaremos PyTorch pela sua flexibilidade.

**Instalação:**
```bash
sudo pip3 install torch numpy cma
```

### 2.1. Treino do VAE (Componente V)

O objetivo é aprender uma representação latente `z` do nosso vetor de estado.

**Passo 1: Gerar o Dataset**

Execute o seu ambiente com ações aleatórias por um longo período (ex: 1 milhão de passos) e guarde cada `state_vector` num ficheiro.

```python
# generate_vae_data.py
import gymnasium as gym
import numpy as np

N_SAMPLES = 1_000_000
dataset = []

env = gym.make('Reacher-v4', xml_file='robot_world.xml')
obs, _ = env.reset()

for i in range(N_SAMPLES):
    action = env.action_space.sample()
    obs, _, terminated, truncated, _ = env.step(action)
    
    joint_angles = obs[:3] # Ajuste conforme a sua observação
    object_pos = env.data.get_body_xpos('object')
    target_pos = env.data.get_site_xpos('target')
    state_vector = np.concatenate([joint_angles, object_pos, target_pos])
    dataset.append(state_vector)

    if (i + 1) % 10000 == 0:
        print(f"Generated {i+1}/{N_SAMPLES} samples")
    if terminated or truncated:
        obs, _ = env.reset()

np.save('vae_dataset.npy', np.array(dataset))
env.close()
```

**Passo 2: Definir e Treinar o Modelo VAE**

```python
# train_vae.py
import torch
import torch.nn as nn
import numpy as np

STATE_DIM = 9 # 3 juntas + 3 obj_pos + 3 target_pos
LATENT_DIM = 32

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, STATE_DIM)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- Loop de Treino ---
dataset = torch.from_numpy(np.load('vae_dataset.npy')).float()
model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    recon_batch, mu, logvar = model(dataset)
    loss = loss_function(recon_batch, dataset, mu, logvar)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item() / len(dataset)}')

# Salvar o modelo treinado
torch.save(model.state_dict(), 'vae.pth')
```

### 2.2. Treino do MDN-RNN (Componente M)

O objetivo é prever o próximo estado latente `z_{t+1}` dado o estado e ação atuais `(z_t, a_t)`.

**Passo 1: Gerar o Dataset de Séries Temporais**

Similar ao VAE, mas agora guardamos sequências de `(z, action, next_z)`.

```python
# generate_rnn_data.py
import gymnasium as gym
import numpy as np
import torch

# Carregar o VAE treinado
vae = VAE()
vae.load_state_dict(torch.load('vae.pth'))
vae.eval()

N_SEQUENCES = 10000
SEQUENCE_LEN = 100
dataset = []

env = gym.make('Reacher-v4', xml_file='robot_world.xml')

for i in range(N_SEQUENCES):
    obs, _ = env.reset()
    sequence = []
    for t in range(SEQUENCE_LEN):
        action = env.action_space.sample()
        
        # Estado atual para z_t
        state_vector = np.concatenate([obs[:3], env.data.get_body_xpos('object'), env.data.get_site_xpos('target')])
        with torch.no_grad():
            mu, _ = vae.encode(torch.from_numpy(state_vector).float())
        z = mu.numpy()

        # Step
        obs, _, terminated, truncated, _ = env.step(action)
        
        sequence.append((z, action))
        if terminated or truncated:
            break
    dataset.append(sequence)

np.save('rnn_dataset.npy', dataset, allow_pickle=True)
env.close()
```

**Passo 2: Definir e Treinar o Modelo MDN-RNN**

Este modelo é mais complexo. Ele combina um LSTM com uma Mixture Density Network.

```python
# train_rnn.py
# (A implementação completa de um MDN-RNN é extensa)
# Recomendo usar uma biblioteca como a de 'hardmaru/pytorch_mdn' no GitHub
# O conceito é:

class MDNRNN(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size, n_gaussians):
        super(MDNRNN, self).__init__()
        self.lstm = nn.LSTM(latent_size + action_size, hidden_size)
        # A cabeça da MDN prevê os parâmetros (pi, sigma, mu) para uma mistura de Gaussianas
        self.mdn_head = nn.Linear(hidden_size, n_gaussians * (2 * latent_size + 1))

    def forward(self, z, a, h):
        x = torch.cat([z, a], dim=1)
        out, h_next = self.lstm(x.unsqueeze(0), h)
        # ... lógica para extrair pi, sigma, mu da saída da mdn_head
        return (pi, sigma, mu), h_next

# O loop de treino minimizaria a negative log-likelihood da verdadeira sequência z_{t+1}
# sob a distribuição de mistura prevista pelo modelo.
# torch.save(mdnrnn.state_dict(), 'mdnrnn.pth')
```

### 2.3. Treino do Controlador (Componente C)

Esta é a parte mais "mágica". Usamos um otimizador de caixa-preta (CMA-ES) para treinar o controlador **dentro do sonho do World Model**.

```python
# train_controller.py
import cma
import torch

# Carregar modelos VAE e MDN-RNN
vae = VAE(); vae.load_state_dict(torch.load('vae.pth'))
mdnrnn = MDNRNN(...); mdnrnn.load_state_dict(torch.load('mdnrnn.pth'))

# Controlador é uma rede simples
controller = nn.Linear(LATENT_DIM + HIDDEN_SIZE, ACTION_DIM)

# Função de fitness: avalia um controlador num "sonho"
def evaluate_controller(weights):
    controller.load_state_dict(weights) # Carregar os pesos propostos pelo CMA-ES
    total_reward = 0

    # Iniciar um sonho
    z = torch.randn(1, LATENT_DIM)
    h = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))

    for _ in range(200): # Sonhar por 200 passos
        # 1. Obter ação do controlador
        action = controller(torch.cat([z, h[0].squeeze(0)], dim=1))
        
        # 2. Prever o próximo estado com o RNN
        (pi, sigma, mu), h = mdnrnn(z, action, h)
        # ... lógica para amostrar o próximo z da mistura de Gaussianas
        z = sample_from_mixture(pi, sigma, mu)

        # 3. Calcular a recompensa decodificando o estado
        decoded_state = vae.decode(z)
        object_pos = decoded_state[0, 3:6]
        target_pos = decoded_state[0, 6:9]
        reward = -torch.norm(object_pos - target_pos) # Recompensa é a distância negativa
        total_reward += reward

    return -total_reward # CMA-ES minimiza, então retornamos a recompensa negativa

# Otimização com CMA-ES
initial_weights = controller.state_dict()
es = cma.CMAEvolutionStrategy(initial_weights_vector, 0.5)

while not es.stop():
    solutions = es.ask()
    fitness_list = [evaluate_controller(s) for s in solutions]
    es.tell(solutions, fitness_list)
    es.logger.add()
    es.disp()

# Salvar o melhor controlador encontrado
# torch.save(es.result.xbest, 'controller.pth')
```

## 3. Executando o Agente Final

Depois de treinar todos os componentes, o loop final de execução no ambiente real (simulado) é surpreendentemente simples:

```python
# run_agent.py

# Carregar todos os modelos treinados (VAE, MDNRNN, Controller)

obs, _ = env.reset()
h = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))

while True:
    # 1. Observar o estado real e codificar para o espaço latente
    state_vector = ... # Extrair do obs
    z, _ = vae.encode(torch.from_numpy(state_vector).float())

    # 2. Obter a ação do controlador
    action_tensor = controller(torch.cat([z, h[0].squeeze(0)], dim=1))
    action = action_tensor.detach().numpy()

    # 3. Executar a ação no ambiente
    obs, _, _, _, _ = env.step(action)

    # 4. Atualizar o estado da memória (opcional, mas bom para consistência)
    _, h = mdnrnn(z, action_tensor, h)
```

Este guia fornece a estrutura completa. A implementação de detalhes como a amostragem da MDN e a interface com o CMA-ES requerem um pouco mais de código, mas a lógica central está toda aqui. Comece por aqui, e terá um World Model funcional em pouco tempo!


## 4. Dicas Práticas e Troubleshooting

### 4.1. Acelerar o Treino

**Paralelização:** Gere dados de treino em paralelo usando múltiplos processos. O MuJoCo é thread-safe e pode correr múltiplas instâncias simultaneamente.

```python
from multiprocessing import Pool

def collect_data(seed):
    env = gym.make('Reacher-v4', xml_file='robot_world.xml')
    env.seed(seed)
    # ... coletar dados
    return data

with Pool(8) as p:
    all_data = p.map(collect_data, range(8))
```

**GPU:** Use GPU para treinar o VAE e o MDN-RNN. Mova os modelos e dados para CUDA:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
dataset = dataset.to(device)
```

### 4.2. Debugging Comum

**Problema:** O VAE não reconstrói bem o estado.

**Solução:** Normalize os seus dados! Diferentes features (ângulos vs posições) têm escalas diferentes.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dataset_normalized = scaler.fit_transform(dataset)
# Guarde o scaler para usar na inferência!
```

**Problema:** O controlador não aprende nada com CMA-ES.

**Solução:** A função de recompensa pode estar mal definida. Certifique-se de que:
- A recompensa é densa (não esparsa)
- O range de valores é razoável (ex: entre -100 e 0)
- Teste primeiro com um controlador aleatório para ver o baseline

**Problema:** O MDN-RNN prevê sempre o mesmo estado.

**Solução:** Isto é "mode collapse". Aumente o número de componentes Gaussianas na mistura (ex: de 5 para 10) e adicione um termo de regularização para encorajar diversidade.

### 4.3. Validação do Sistema

Antes de considerar o treino completo, valide cada componente isoladamente:

1. **VAE:** Visualize reconstruções. Pegue em 10 estados aleatórios, passe pelo VAE, e compare o input com o output. Devem ser quase idênticos.

2. **MDN-RNN:** Teste a predição. Dê-lhe uma sequência real `(z_t, a_t)` e veja se `z_{t+1}` previsto está perto do `z_{t+1}` real. Calcule o MSE.

3. **Controlador:** Execute-o no ambiente real (não no sonho) e veja se consegue pelo menos aproximar-se do alvo, mesmo que de forma ineficiente.

### 4.4. Estrutura de Ficheiros Recomendada

```
project/
├── data/
│   ├── vae_dataset.npy
│   └── rnn_dataset.npy
├── models/
│   ├── vae.pth
│   ├── mdnrnn.pth
│   └── controller.pth
├── configs/
│   └── robot_world.xml
├── src/
│   ├── train_vae.py
│   ├── train_rnn.py
│   ├── train_controller.py
│   └── run_agent.py
└── requirements.txt
```

### 4.5. Próximos Passos

Depois de ter a Fase 1 a funcionar:

1. **Adicione métricas:** Taxa de sucesso, distância média ao alvo, suavidade da trajetória
2. **Experimente com hiperparâmetros:** Tamanho do espaço latente, número de componentes da MDN
3. **Prepare-se para a Fase 2:** Comece a pensar em como representar restrições estruturadamente

Este é o caminho. Boa sorte, e lembre-se: a primeira versão não será perfeita. Itere, experimente, e aprenda com os erros! 🚀
