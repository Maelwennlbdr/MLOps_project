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
    "MLFLOW_TRACKING_URI", "https://dagshub.com/louiseLV/MLOps_project-dagshub.mlflow"
)
MODEL_NAME = "mlops-model"
ACCURACY_THRESHOLD = 0.74  # Seuil minimum d'accuracy

# Seuils de quality gates
QUALITY_GATES = {
    "accuracy": ACCURACY_THRESHOLD,
}
EXPERIMENT_NAME = "diabetes-mlops-experiment"


def check_quality_gates():
    """
    Vérifie les quality gates de la dernière version du modèle.

    Returns:
        bool: True si le modèle passe tous les tests, False sinon
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Récupérer le modèle avec alias "Staging" (MLflow 2.9.0+)
        registered_model = client.get_registered_model(MODEL_NAME)

        # Chercher la version avec l'alias "Staging"
        # MLflow expose aliases comme un dict alias -> version dans les versions récentes.
        staging_version = None
        aliases = registered_model.aliases

        if isinstance(aliases, dict):
            staging_version = aliases.get("Staging")
        else:
            # Fallback défensif pour d'anciens formats éventuels.
            for alias_info in aliases or []:
                if getattr(alias_info, "alias", None) == "Staging":
                    staging_version = getattr(alias_info, "version", None)
                    break

        if not staging_version:
            print(f"❌ Aucun modèle trouvé avec l'alias 'Staging'")
            print(f"💡 Exécutez 'python ml/train.py' pour créer un modèle entraîné")
            return False

        print(f"📊 Vérification des quality gates pour {MODEL_NAME} v{staging_version}")
        print("-" * 60)

        # Récupérer les runs pour cette version du modèle
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"⚠️  Expérience MLflow introuvable: {EXPERIMENT_NAME}")
            return False

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=5,
        )

        if not runs:
            print(f"⚠️  Aucun run trouvé dans l'expérience")
            return False

        # Trouver le run correspondant à cette version
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

            # Promouvoir en Production avec l'alias moderne
            promote_to_production(client, MODEL_NAME, staging_version)
            return True
        else:
            print("\n" + "=" * 60)
            print("❌ AU MOINS UN QUALITY GATE A ÉCHOUÉ")
            print(
                f"Modèle {MODEL_NAME} v{staging_version} reste avec l'alias 'Staging'"
            )
            print("=" * 60)
            return False

    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        import traceback

        traceback.print_exc()
        return False


def promote_to_production(client, model_name, version):
    """
    Promeut le modèle en alias 'Production' (MLflow 2.9.0+)
    """
    try:
        # Utiliser l'alias API moderne au lieu des stages
        client.set_registered_model_alias(
            name=model_name, alias="Production", version=version
        )
        print(f"\n🚀 Modèle {model_name} v{version} promu avec l'alias 'Production'!")
    except Exception as e:
        print(f"❌ Erreur lors de la promotion: {e}")
        raise


if __name__ == "__main__":
    success = check_quality_gates()
    sys.exit(0 if success else 1)
