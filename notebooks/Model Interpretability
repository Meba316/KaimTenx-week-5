pip install shap lime
import shap

explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(validation_dataset['input_ids'])

shap.summary_plot(shap_values, validation_dataset['input_ids'])

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=model.config.id2label.values())
explanation = explainer.explain_instance(validation_text, model.predict_proba)
explanation.show_in_notebook()
