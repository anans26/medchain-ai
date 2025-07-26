import React, { useState } from 'react'
import { Brain, Send, AlertCircle, CheckCircle, Clock, Shield } from 'lucide-react'

const DiagnosisInterface = () => {
  const [patientData, setPatientData] = useState({
    patientId: '',
    symptoms: [],
    medicalHistory: []
  })
  const [newSymptom, setNewSymptom] = useState('')
  const [newHistory, setNewHistory] = useState('')
  const [diagnosis, setDiagnosis] = useState(null)
  const [loading, setLoading] = useState(false)

  const commonSymptoms = [
    'fever', 'cough', 'headache', 'fatigue', 'chest_pain', 
    'shortness_of_breath', 'nausea', 'dizziness', 'joint_pain'
  ]

  const addSymptom = (symptom) => {
    if (symptom && !patientData.symptoms.includes(symptom)) {
      setPatientData(prev => ({
        ...prev,
        symptoms: [...prev.symptoms, symptom]
      }))
      setNewSymptom('')
    }
  }

  const addHistory = (history) => {
    if (history && !patientData.medicalHistory.includes(history)) {
      setPatientData(prev => ({
        ...prev,
        medicalHistory: [...prev.medicalHistory, history]
      }))
      setNewHistory('')
    }
  }

  const removeSymptom = (symptom) => {
    setPatientData(prev => ({
      ...prev,
      symptoms: prev.symptoms.filter(s => s !== symptom)
    }))
  }

  const removeHistory = (history) => {
    setPatientData(prev => ({
      ...prev,
      medicalHistory: prev.medicalHistory.filter(h => h !== history)
    }))
  }

  const submitDiagnosis = async () => {
    if (!patientData.patientId || patientData.symptoms.length === 0) {
      alert('Please provide patient ID and at least one symptom')
      return
    }

    setLoading(true)
    
    // Simulate API call to AI inference canister
    try {
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Mock diagnosis result
      const mockDiagnosis = {
        diagnosis: patientData.symptoms.includes('fever') && patientData.symptoms.includes('cough') 
          ? 'Possible respiratory infection' 
          : 'General consultation recommended',
        confidence: 0.85,
        recommendations: [
          'Consult with healthcare provider',
          'Monitor symptoms closely',
          'Follow up in 48 hours if symptoms persist'
        ],
        riskFactors: ['Age', 'Medical history'],
        modelVersion: 'v1.2.3',
        signature: 'verified'
      }
      
      setDiagnosis(mockDiagnosis)
    } catch (error) {
      console.error('Diagnosis failed:', error)
      alert('Diagnosis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="text-white">
      <div className="flex items-center mb-6">
        <Brain className="w-8 h-8 mr-3 text-blue-400" />
        <h2 className="text-2xl font-bold">AI-Powered Medical Diagnosis</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium mb-2">Patient ID</label>
            <input
              type="text"
              value={patientData.patientId}
              onChange={(e) => setPatientData(prev => ({ ...prev, patientId: e.target.value }))}
              className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
              placeholder="Enter patient identifier"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Symptoms</label>
            <div className="flex mb-3">
              <input
                type="text"
                value={newSymptom}
                onChange={(e) => setNewSymptom(e.target.value)}
                className="flex-1 px-4 py-2 bg-white/10 border border-white/20 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
                placeholder="Add symptom"
                onKeyPress={(e) => e.key === 'Enter' && addSymptom(newSymptom)}
              />
              <button
                onClick={() => addSymptom(newSymptom)}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-r-lg transition-colors"
              >
                Add
              </button>
            </div>
            
            <div className="flex flex-wrap gap-2 mb-3">
              {commonSymptoms.map(symptom => (
                <button
                  key={symptom}
                  onClick={() => addSymptom(symptom)}
                  className="px-3 py-1 text-xs bg-white/10 hover:bg-white/20 rounded-full transition-colors"
                >
                  {symptom.replace('_', ' ')}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap gap-2">
              {patientData.symptoms.map(symptom => (
                <span
                  key={symptom}
                  className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm flex items-center"
                >
                  {symptom.replace('_', ' ')}
                  <button
                    onClick={() => removeSymptom(symptom)}
                    className="ml-2 text-blue-300 hover:text-white"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Medical History</label>
            <div className="flex mb-3">
              <input
                type="text"
                value={newHistory}
                onChange={(e) => setNewHistory(e.target.value)}
                className="flex-1 px-4 py-2 bg-white/10 border border-white/20 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
                placeholder="Add medical history"
                onKeyPress={(e) => e.key === 'Enter' && addHistory(newHistory)}
              />
              <button
                onClick={() => addHistory(newHistory)}
                className="px-4 py-2 bg-green-500 hover:bg-green-600 rounded-r-lg transition-colors"
              >
                Add
              </button>
            </div>

            <div className="flex flex-wrap gap-2">
              {patientData.medicalHistory.map(history => (
                <span
                  key={history}
                  className="px-3 py-1 bg-green-500/20 text-green-300 rounded-full text-sm flex items-center"
                >
                  {history}
                  <button
                    onClick={() => removeHistory(history)}
                    className="ml-2 text-green-300 hover:text-white"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          <button
            onClick={submitDiagnosis}
            disabled={loading || !patientData.patientId || patientData.symptoms.length === 0}
            className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium flex items-center justify-center transition-all"
          >
            {loading ? (
              <>
                <Clock className="w-5 h-5 mr-2 animate-spin" />
                Processing on-chain...
              </>
            ) : (
              <>
                <Send className="w-5 h-5 mr-2" />
                Get AI Diagnosis
              </>
            )}
          </button>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {diagnosis && (
            <div className="bg-white/5 rounded-lg p-6 border border-white/10">
              <div className="flex items-center mb-4">
                <CheckCircle className="w-6 h-6 text-green-400 mr-2" />
                <h3 className="text-xl font-semibold">Diagnosis Result</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="font-medium text-blue-300 mb-2">Primary Diagnosis</h4>
                  <p className="text-lg">{diagnosis.diagnosis}</p>
                </div>

                <div>
                  <h4 className="font-medium text-blue-300 mb-2">Confidence Score</h4>
                  <div className="flex items-center">
                    <div className="flex-1 bg-white/10 rounded-full h-2 mr-3">
                      <div 
                        className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full"
                        style={{ width: `${diagnosis.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm">{(diagnosis.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium text-blue-300 mb-2">Recommendations</h4>
                  <ul className="space-y-1">
                    {diagnosis.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start">
                        <AlertCircle className="w-4 h-4 text-yellow-400 mr-2 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between text-sm text-white/60">
                    <div className="flex items-center">
                      <Shield className="w-4 h-4 mr-1" />
                      Model: {diagnosis.modelVersion}
                    </div>
                    <div className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-1 text-green-400" />
                      Signature Verified
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold mb-4">Privacy & Security</h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-center">
                <Shield className="w-4 h-4 text-green-400 mr-2" />
                <span>Data processed with differential privacy</span>
              </div>
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 text-green-400 mr-2" />
                <span>Results signed with threshold-ECDSA</span>
              </div>
              <div className="flex items-center">
                <Brain className="w-4 h-4 text-blue-400 mr-2" />
                <span>AI inference runs on-chain in Rust canisters</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DiagnosisInterface