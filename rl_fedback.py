"""
===============================================================================
🎯 RLHF MODULE - REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
===============================================================================

Implementação completa de RLHF para o Pipeline AGI Generativa v2.0

Componentes:
1. Reward Model (aprende preferências humanas)
2. Feedback Collector (coleta e processa feedback)
3. Policy Updater (atualiza modelos com base no feedback)
4. Preference Ranker (ranking de respostas)

Autor: João Manoel
Referência: InstructGPT (OpenAI), Constitutional AI (Anthropic)
===============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# 1. REWARD MODEL
# ============================================

class RewardModel:
    """
    Modelo de recompensa que aprende preferências humanas
    
    Treina um classificador para prever qual resposta é melhor
    baseado em pares de comparação (A vs B, qual é melhor?)
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = model_path or Path("models/reward_model.pkl")
        self.feature_dim = 20  # Dimensão das features das respostas
        self.is_trained = False
        
        # Carregar modelo existente se disponível
        if self.model_path.exists():
            self.load()
    
    def _extract_features(self, response_data: Dict) -> np.ndarray:
        """
        Extrai features de uma resposta para o reward model
        
        Features:
        - Confiança da predição
        - Severidade detectada
        - Número de causas identificadas
        - Tamanho da explicação
        - Urgência da ação recomendada
        - etc.
        """
        features = []
        
        # 1. Features de predição
        prediction = response_data.get('prediction', {})
        features.append(prediction.get('value', 0))
        features.append(prediction.get('confidence', 0))
        
        # 2. Features de raciocínio
        reasoning = response_data.get('reasoning', {})
        severity_map = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'CRITICAL': 1.0}
        features.append(severity_map.get(reasoning.get('severity', 'LOW'), 0))
        causes = reasoning.get('identified_causes', [])
        features.append(len(causes))
        features.append(1.0 if len(causes) > 0 else 0.0)
        
        # 3. Features de decisão
        decision = response_data.get('decision', {})
        priority_map = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'URGENT': 1.0}
        features.append(priority_map.get(decision.get('priority', 'LOW'), 0))
        plan = decision.get('execution_plan', [])
        features.append(len(plan))
        
        # 4. Features de explicação
        explanation = response_data.get('generated_explanation', '')
        features.append(len(explanation) / 1000)  # Normalizado
        features.append(1.0 if explanation else 0.0)
        
        # 5. Features adicionais (padding para 20 features)
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim])
    
    def train(self, comparison_data: List[Dict], epochs: int = 10):
        """
        Treina o reward model com dados de comparação
        
        comparison_data formato:
        [
            {
                "response_a": {...},
                "response_b": {...},
                "preferred": "a" or "b",
                "score_diff": float  # opcional
            }
        ]
        """
        logger.info(f"🎓 Treinando Reward Model com {len(comparison_data)} comparações...")
        
        X_train = []
        y_train = []
        
        for comp in comparison_data:
            # Extrair features de ambas as respostas
            features_a = self._extract_features(comp['response_a'])
            features_b = self._extract_features(comp['response_b'])
            
            # Criar par de features (diferença)
            feature_diff = features_a - features_b
            X_train.append(feature_diff)
            
            # Label: 1 se A é preferido, 0 se B é preferido
            y_train.append(1 if comp['preferred'] == 'a' else 0)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Treinar modelo
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Avaliar
        y_pred = self.model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        logger.info(f"✅ Reward Model treinado! Acurácia: {accuracy:.2%}")
        
        # Salvar
        self.save()
        
        return accuracy
    
    def predict_reward(self, response_data: Dict) -> float:
        """
        Prediz a recompensa (score) de uma resposta
        
        Retorna: float entre 0 e 1
        """
        if not self.is_trained or self.model is None:
            # Se não treinado, usar heurística simples
            return self._heuristic_reward(response_data)
        
        features = self._extract_features(response_data)
        
        # Usar probabilidade da classe positiva como reward
        reward = self.model.predict_proba(features.reshape(1, -1))[0][1]
        
        return float(reward)
    
    def _heuristic_reward(self, response_data: Dict) -> float:
        """
        Recompensa heurística quando modelo não está treinado
        """
        score = 0.5  # Base
        
        # Bonificar alta confiança
        confidence = response_data.get('prediction', {}).get('confidence', 0)
        score += confidence * 0.2
        
        # Bonificar identificação de causas
        causes = response_data.get('reasoning', {}).get('identified_causes', [])
        score += min(len(causes) * 0.1, 0.2)
        
        # Bonificar plano de ação
        plan = response_data.get('decision', {}).get('execution_plan', [])
        score += min(len(plan) * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def compare_responses(self, response_a: Dict, response_b: Dict) -> str:
        """
        Compara duas respostas e retorna qual é melhor
        
        Retorna: 'a' ou 'b'
        """
        reward_a = self.predict_reward(response_a)
        reward_b = self.predict_reward(response_b)
        
        return 'a' if reward_a > reward_b else 'b'
    
    def save(self):
        """Salvar modelo"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_dim': self.feature_dim,
            'is_trained': self.is_trained
        }, self.model_path)
        logger.info(f"💾 Reward Model salvo em {self.model_path}")
    
    def load(self):
        """Carregar modelo"""
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_dim = data['feature_dim']
            self.is_trained = data['is_trained']
            logger.info(f"✅ Reward Model carregado de {self.model_path}")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar Reward Model: {e}")

# ============================================
# 2. FEEDBACK COLLECTOR
# ============================================

class FeedbackCollector:
    """
    Coleta e processa feedback humano
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/feedback.jsonl")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_buffer = []
    
    def add_feedback(self, 
                     result_id: str,
                     response_data: Dict,
                     score: float,
                     feedback_text: str = "",
                     user_id: str = "anonymous"):
        """
        Adiciona feedback à coleção
        """
        feedback_entry = {
            "result_id": result_id,
            "response_data": response_data,
            "score": score,
            "feedback_text": feedback_text,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Salvar em disco
        self._append_to_file(feedback_entry)
        
        logger.info(f"💬 Feedback adicionado: {result_id} (score={score:.2f})")
    
    def _append_to_file(self, feedback_entry: Dict):
        """Append feedback ao arquivo JSONL"""
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
    
    def load_all_feedback(self) -> List[Dict]:
        """Carregar todo o feedback do arquivo"""
        if not self.storage_path.exists():
            return []
        
        feedback_list = []
        with open(self.storage_path, 'r') as f:
            for line in f:
                try:
                    feedback_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"📂 {len(feedback_list)} feedbacks carregados")
        return feedback_list
    
    def get_recent_feedback(self, days: int = 7) -> List[Dict]:
        """Obter feedback recente"""
        all_feedback = self.load_all_feedback()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent = [
            fb for fb in all_feedback
            if datetime.fromisoformat(fb['timestamp']) > cutoff_date
        ]
        
        return recent
    
    def create_comparison_pairs(self, min_score_diff: float = 0.2) -> List[Dict]:
        """
        Cria pares de comparação para treinar o reward model
        
        Pega feedback com scores diferentes e cria pares (melhor vs pior)
        """
        all_feedback = self.load_all_feedback()
        
        if len(all_feedback) < 2:
            logger.warning("⚠️ Poucos feedbacks para criar pares")
            return []
        
        # Ordenar por score
        sorted_feedback = sorted(all_feedback, key=lambda x: x['score'], reverse=True)
        
        comparison_pairs = []
        
        # Criar pares: alto score vs baixo score
        n_pairs = min(len(sorted_feedback) // 2, 100)  # Limitar a 100 pares
        
        for i in range(n_pairs):
            high_score_fb = sorted_feedback[i]
            low_score_fb = sorted_feedback[-(i+1)]
            
            score_diff = high_score_fb['score'] - low_score_fb['score']
            
            if score_diff >= min_score_diff:
                comparison_pairs.append({
                    'response_a': high_score_fb['response_data'],
                    'response_b': low_score_fb['response_data'],
                    'preferred': 'a',
                    'score_diff': score_diff
                })
        
        logger.info(f"🔗 {len(comparison_pairs)} pares de comparação criados")
        return comparison_pairs
    
    def get_statistics(self) -> Dict:
        """Estatísticas do feedback"""
        all_feedback = self.load_all_feedback()
        
        if not all_feedback:
            return {
                "total": 0,
                "average_score": 0,
                "score_distribution": {}
            }
        
        scores = [fb['score'] for fb in all_feedback]
        
        return {
            "total": len(all_feedback),
            "average_score": np.mean(scores),
            "median_score": np.median(scores),
            "std_score": np.std(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_distribution": {
                "excellent (>0.8)": sum(1 for s in scores if s > 0.8),
                "good (0.6-0.8)": sum(1 for s in scores if 0.6 <= s <= 0.8),
                "fair (0.4-0.6)": sum(1 for s in scores if 0.4 <= s < 0.6),
                "poor (<0.4)": sum(1 for s in scores if s < 0.4)
            }
        }

# ============================================
# 3. POLICY UPDATER
# ============================================

class PolicyUpdater:
    """
    Atualiza a política (modelos) com base no feedback
    
    Em um sistema completo, isso ajustaria os pesos do modelo generativo
    Aqui, vamos ajustar os thresholds de decisão
    """
    
    def __init__(self):
        self.current_policy = self._get_default_policy()
        self.policy_history = []
    
    def _get_default_policy(self) -> Dict:
        """Política padrão"""
        return {
            "severity_thresholds": {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.0
            },
            "confidence_threshold": 0.6,
            "min_causes_to_report": 1,
            "explanation_min_length": 200,
            "version": 1,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_policy(self, feedback_stats: Dict, reward_model: RewardModel):
        """
        Atualiza política baseado em estatísticas de feedback
        """
        logger.info("🔄 Atualizando política baseado em feedback...")
        
        avg_score = feedback_stats.get('average_score', 0.5)
        
        # Salvar política antiga
        self.policy_history.append(self.current_policy.copy())
        
        # Ajustar thresholds baseado no feedback
        if avg_score < 0.5:
            # Feedback ruim: tornar sistema mais conservador
            logger.info("📉 Feedback abaixo da média, ajustando para ser mais conservador")
            self.current_policy['severity_thresholds']['critical'] = 0.85
            self.current_policy['severity_thresholds']['high'] = 0.65
            self.current_policy['confidence_threshold'] = 0.65
        
        elif avg_score > 0.8:
            # Feedback bom: manter ou relaxar um pouco
            logger.info("📈 Feedback excelente, mantendo política atual")
            # Não mudar muito quando está bom
        
        else:
            # Feedback médio: ajuste moderado
            logger.info("📊 Feedback médio, ajuste fino")
            self.current_policy['severity_thresholds']['high'] = 0.7
        
        # Atualizar versão
        self.current_policy['version'] += 1
        self.current_policy['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"✅ Política atualizada para v{self.current_policy['version']}")
        
        return self.current_policy
    
    def get_current_policy(self) -> Dict:
        """Obter política atual"""
        return self.current_policy
    
    def save_policy(self, path: Path):
        """Salvar política"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'current': self.current_policy,
                'history': self.policy_history
            }, f, indent=2)
        logger.info(f"💾 Política salva em {path}")
    
    def load_policy(self, path: Path):
        """Carregar política"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.current_policy = data['current']
            self.policy_history = data.get('history', [])
            logger.info(f"✅ Política carregada de {path}")
        except Exception as e:
            logger.warning(f"⚠️ Erro ao carregar política: {e}")

# ============================================
# 4. RLHF MANAGER (ORQUESTRADOR)
# ============================================

class RLHFManager:
    """
    Gerenciador principal do sistema RLHF
    
    Coordena: Reward Model, Feedback Collector, Policy Updater
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("rlhf_data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.reward_model = RewardModel(self.base_path / "reward_model.pkl")
        self.feedback_collector = FeedbackCollector(self.base_path / "feedback.jsonl")
        self.policy_updater = PolicyUpdater()
        
        logger.info("✅ RLHF Manager inicializado")
    
    def collect_feedback(self, 
                        result_id: str,
                        response_data: Dict,
                        score: float,
                        feedback_text: str = "",
                        user_id: str = "anonymous"):
        """
        Coletar feedback de uma resposta
        """
        self.feedback_collector.add_feedback(
            result_id, response_data, score, feedback_text, user_id
        )
    
    def train_reward_model(self, min_feedback_count: int = 10):
        """
        Treinar reward model com feedback acumulado
        """
        logger.info("🎓 Iniciando treinamento do Reward Model...")
        
        # Verificar se há feedback suficiente
        stats = self.feedback_collector.get_statistics()
        if stats['total'] < min_feedback_count:
            logger.warning(f"⚠️ Apenas {stats['total']} feedbacks. Mínimo: {min_feedback_count}")
            return None
        
        # Criar pares de comparação
        comparison_pairs = self.feedback_collector.create_comparison_pairs()
        
        if len(comparison_pairs) < 5:
            logger.warning("⚠️ Poucos pares de comparação criados")
            return None
        
        # Treinar
        accuracy = self.reward_model.train(comparison_pairs)
        
        return accuracy
    
    def update_system_policy(self):
        """
        Atualizar política do sistema baseado em feedback
        """
        logger.info("🔄 Atualizando política do sistema...")
        
        stats = self.feedback_collector.get_statistics()
        updated_policy = self.policy_updater.update_policy(stats, self.reward_model)
        
        # Salvar
        self.policy_updater.save_policy(self.base_path / "policy.json")
        
        return updated_policy
    
    def get_system_status(self) -> Dict:
        """
        Status completo do sistema RLHF
        """
        feedback_stats = self.feedback_collector.get_statistics()
        current_policy = self.policy_updater.get_current_policy()
        
        return {
            "reward_model": {
                "trained": self.reward_model.is_trained,
                "model_path": str(self.reward_model.model_path)
            },
            "feedback": feedback_stats,
            "policy": {
                "version": current_policy['version'],
                "last_updated": current_policy['last_updated']
            }
        }
    
    def run_rlhf_cycle(self, min_feedback: int = 10):
        """
        Executar um ciclo completo de RLHF
        
        1. Coletar feedback (já feito continuamente)
        2. Treinar reward model
        3. Atualizar política
        """
        logger.info("🔄 Executando ciclo RLHF...")
        
        # 1. Treinar reward model
        accuracy = self.train_reward_model(min_feedback)
        
        if accuracy is None:
            logger.warning("⚠️ Não foi possível treinar reward model")
            return {"status": "insufficient_data"}
        
        # 2. Atualizar política
        updated_policy = self.update_system_policy()
        
        # 3. Resultado
        result = {
            "status": "success",
            "reward_model_accuracy": accuracy,
            "policy_version": updated_policy['version'],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Ciclo RLHF concluído: accuracy={accuracy:.2%}")
        
        return result

# ============================================
# 5. FUNÇÕES AUXILIARES
# ============================================

def simulate_feedback_data(n_samples: int = 50) -> List[Dict]:
    """
    Simula dados de feedback para testes
    """
    logger.info(f"🔧 Gerando {n_samples} feedbacks sintéticos...")
    
    feedback_list = []
    
    for i in range(n_samples):
        # Simular resposta
        confidence = np.random.uniform(0.5, 1.0)
        severity = np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        
        response_data = {
            'prediction': {
                'value': np.random.uniform(0, 1),
                'confidence': confidence
            },
            'reasoning': {
                'severity': severity,
                'identified_causes': [f"Causa {j}" for j in range(np.random.randint(1, 4))]
            },
            'decision': {
                'priority': severity,
                'execution_plan': [f"Passo {j}" for j in range(np.random.randint(2, 6))]
            },
            'generated_explanation': "Explicação " * np.random.randint(20, 50)
        }
        
        # Score correlacionado com qualidade
        base_score = confidence * 0.4
        severity_bonus = {'LOW': 0.1, 'MEDIUM': 0.2, 'HIGH': 0.3, 'CRITICAL': 0.4}[severity]
        score = min(base_score + severity_bonus + np.random.uniform(-0.1, 0.1), 1.0)
        
        feedback_list.append({
            'result_id': f"sim_{i}",
            'response_data': response_data,
            'score': score,
            'feedback_text': f"Feedback simulado {i}",
            'user_id': f"user_{i % 10}",
            'timestamp': datetime.now().isoformat()
        })
    
    return feedback_list

# ============================================
# 6. EXEMPLO DE USO
# ============================================

def main():
    """
    Exemplo de uso do sistema RLHF
    """
    print("="*80)
    print("🎯 RLHF SYSTEM - EXEMPLO DE USO")
    print("="*80)
    
    # Inicializar manager
    rlhf = RLHFManager(Path("rlhf_data_test"))
    
    # Simular coleta de feedback
    print("\n1️⃣ Simulando coleta de feedback...")
    simulated_feedback = simulate_feedback_data(50)
    
    for fb in simulated_feedback:
        rlhf.collect_feedback(
            fb['result_id'],
            fb['response_data'],
            fb['score'],
            fb['feedback_text'],
            fb['user_id']
        )
    
    # Ver estatísticas
    print("\n2️⃣ Estatísticas de feedback:")
    stats = rlhf.feedback_collector.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Executar ciclo RLHF
    print("\n3️⃣ Executando ciclo RLHF...")
    result = rlhf.run_rlhf_cycle(min_feedback=10)
    print(json.dumps(result, indent=2))
    
    # Status do sistema
    print("\n4️⃣ Status do sistema:")
    status = rlhf.get_system_status()
    print(json.dumps(status, indent=2))
    
    # Testar reward model
    print("\n5️⃣ Testando Reward Model...")
    test_response = simulated_feedback[0]['response_data']
    reward = rlhf.reward_model.predict_reward(test_response)
    print(f"Reward predito: {reward:.3f}")
    
    print("\n" + "="*80)
    print("✅ EXEMPLO CONCLUÍDO!")
    print("="*80)

if __name__ == "__main__":
    main()
