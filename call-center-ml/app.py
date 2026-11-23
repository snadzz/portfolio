{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ee8b32eb-11b4-4aee-8efe-0a81abdb1e22",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save as app.py and run with: uvicorn app:app --reload\n",
    "from fastapi import FastAPI\n",
    "from pydantic import BaseModel\n",
    "import joblib\n",
    "import pandas as pd\n",
    "\n",
    "app = FastAPI()\n",
    "model = joblib.load('call_center_model.pkl')\n",
    "\n",
    "class LeadData(BaseModel):\n",
    "    age: int\n",
    "    race: str\n",
    "    gender: str\n",
    "    city: str\n",
    "    employed: str\n",
    "    mobile: str\n",
    "    qualification: str\n",
    "    agent_rating: int\n",
    "    product: str\n",
    "    call_hour: int\n",
    "\n",
    "@app.post(\"/predict\")\n",
    "def predict_conversion(lead: LeadData):\n",
    "    input_df = pd.DataFrame([lead.dict()])\n",
    "    probability = model.predict_proba(input_df)[0][1]\n",
    "    prediction = int(probability >= 0.5)\n",
    "    return {\n",
    "        \"will_convert\": bool(prediction),\n",
    "        \"conversion_probability\": f\"{round(probability * 100, 1)}%\",\n",
    "        \"action\": \"Prioritize Lead\" if probability > 0.65 else \"Standard Queue\"\n",
    "    }\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
