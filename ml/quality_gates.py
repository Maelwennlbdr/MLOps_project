"""
Quality Gates Script
Valide les metrics du modèle avant promotion en production
"""

import os
import sys
from mlflow.tracking import MlflowClient
import mlflow

# Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", 
    "https://dagshub.com/louiseLV/MLOps_project-dagshub.mlflow"
)
MODEL_NAME = "mlops-model"
ACCURACY_THRESHOLD = 0.74  # Seuil minimum d'accuracy

# Seuils de quality gates
QUALITY_GATES = {
    "accuracy": ACCURACY_THRESHOLD,
}

def check_quality_gates():
    """
    Vérifie les quality gates de la dernière version du modèle.
    
    Returns:
        bool: True si le modèle passe tous les tests, False sinon
    """
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Récupérer le modèle en stage "Staging"
        staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        
        if not staging_versions:
            print(f"❌ Aucun modèle trouvé en stage 'Staging'")
            return False
        
        staging_version = staging_versions[0]
        version_number = staging_version.version
        
        print(f"📊 Vérification des quality gates pour {MODEL_NAME} v{version_number}")
        print("-" * 60)
        
        # Récupérer les runs pour cette version du modèle
        runs = client.search_runs(
            experiment_names=["diabetes-mlops-experiment"],
            filter_string=f"tags.mlflow.log_model.history LIKE '%{version_number}%'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"⚠️  Aucun run trouvé pour la version {version_number}")
            return False
        
        run = runs[0]
        metrics = run.data.metrics
        
        print(f"📈 Metrics du modèle:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # Vérifier les seuils
        all_passed = True
        print("\n🔍 Vérification des seuils:")
        print("-" * 60)
        
        for gate_name, threshold in QUALITY_GATES.items():
            metric_value = metrics.get(gate_name)
            
            if metric_value is None:
                print(f"❌ Métrique '{gate_name}' non trouvée")
                all_passed = False
            elif metric_value >= threshold:
                print(f"✅ {gate_name}: {metric_value:.4f} >= {threshold} ✓")
            else:
                print(f"❌ {gate_name}: {metric_value:.4f} < {threshold} ✗")
                all_passed = False
        
        if all_passed:
            print("\n" + "=" * 60)
            print("✅ TOUS LES QUALITY GATES SONT PASSÉS!")
            print("=" * 60)
            
            # Promouvoir en Production
            promote_to_production(client, MODEL_NAME, version_number)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ AU MOINS UN QUALITY GATE A ÉCHOUÉ")
            print(f"Modèle {MODEL_NAME} v{version_number} reste en stage 'Staging'")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False

def promote_to_production(client, model_name, version):
    """
    Promeut le modèle en stage 'Production'
    """
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=version
        )
        print(f"\n🚀 Modèle {model_name} v{version} promu en 'Production'!")
    except Exception as e:
        print(f"❌ Erreur lors de la promotion: {e}")
        raise

if __name__ == "__main__":
    success = check_quality_gates()
    sys.exit(0 if success else 1)
