import React, { useState, useEffect } from 'react'
import { Network, Users, Upload, Download, Clock, CheckCircle, AlertTriangle } from 'lucide-react'

const FederatedLearning = () => {
  const [currentRound, setCurrentRound] = useState(null)
  const [institutions, setInstitutions] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)

  useEffect(() => {
    // Mock data for current federated learning round
    setCurrentRound({
      roundId: 12345,
      status: 'Open',
      targetParticipants: 5,
      currentParticipants: 3,
      privacyEpsilon: 1.0,
      deadline: Date.now() + 3600000, // 1 hour from now
      updates: []
    })

    // Mock institutions data
    setInstitutions([
      {
        id: 'hospital_1',
        name: 'General Hospital',
        totalContributions: 15,
        privacyBudgetUsed: 2.5,
        lastUpdate: Date.now() - 300000,
        reputationScore: 0.95,
        status: 'active'
      },
      {
        id: 'clinic_2',
        name: 'Medical Clinic',
        totalContributions: 8,
        privacyBudgetUsed: 1.2,
        lastUpdate: Date.now() - 600000,
        reputationScore: 0.88,
        status: 'active'
      },
      {
        id: 'research_3',
        name: 'Research Institute',
        totalContributions: 22,
        privacyBudgetUsed: 4.1,
        lastUpdate: Date.now() - 900000,
        reputationScore: 0.92,
        status: 'pending'
      }
    ])
  }, [])

  const handleFileSelect = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const uploadGradients = async () => {
    if (!selectedFile) return

    setIsUploading(true)
    setUploadProgress(0)

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setIsUploading(false)
          alert('Gradients uploaded successfully with differential privacy!')
          return 100
        }
        return prev + 10
      })
    }, 200)
  }

  const formatTimeRemaining = (deadline) => {
    const remaining = deadline - Date.now()
    const hours = Math.floor(remaining / 3600000)
    const minutes = Math.floor((remaining % 3600000) / 60000)
    return `${hours}h ${minutes}m`
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-green-400'
      case 'pending': return 'text-yellow-400'
      case 'inactive': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  return (
    <div className="text-white">
      <div className="flex items-center mb-6">
        <Network className="w-8 h-8 mr-3 text-purple-400" />
        <h2 className="text-2xl font-bold">Federated Learning Network</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Current Round Status */}
        <div className="lg:col-span-2 space-y-6">
          {currentRound && (
            <div className="bg-white/5 rounded-lg p-6 border border-white/10">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold">Current Training Round</h3>
                <span className="px-3 py-1 bg-green-500/20 text-green-300 rounded-full text-sm">
                  {currentRound.status}
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">
                    {currentRound.currentParticipants}
                  </div>
                  <div className="text-sm text-white/60">Participants</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">
                    {currentRound.targetParticipants}
                  </div>
                  <div className="text-sm text-white/60">Target</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">
                    ε = {currentRound.privacyEpsilon}
                  </div>
                  <div className="text-sm text-white/60">Privacy Budget</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-400">
                    {formatTimeRemaining(currentRound.deadline)}
                  </div>
                  <div className="text-sm text-white/60">Remaining</div>
                </div>
              </div>

              <div className="mb-4">
                <div className="flex justify-between text-sm mb-2">
                  <span>Progress</span>
                  <span>{currentRound.currentParticipants}/{currentRound.targetParticipants}</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(currentRound.currentParticipants / currentRound.targetParticipants) * 100}%` }}
                  />
                </div>
              </div>

              <div className="flex items-center text-sm text-white/60">
                <Clock className="w-4 h-4 mr-2" />
                Round #{currentRound.roundId} • Started 2 hours ago
              </div>
            </div>
          )}

          {/* Upload Interface */}
          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-xl font-semibold mb-4">Contribute Model Updates</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Upload Gradient Updates (with Differential Privacy)
                </label>
                <div className="border-2 border-dashed border-white/20 rounded-lg p-6 text-center">
                  <input
                    type="file"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="gradient-upload"
                    accept=".json,.pkl,.pt"
                  />
                  <label htmlFor="gradient-upload" className="cursor-pointer">
                    <Upload className="w-12 h-12 text-white/40 mx-auto mb-4" />
                    <p className="text-white/60 mb-2">
                      {selectedFile ? selectedFile.name : 'Click to select gradient file'}
                    </p>
                    <p className="text-xs text-white/40">
                      Supports .json, .pkl, .pt files
                    </p>
                  </label>
                </div>
              </div>

              {selectedFile && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2">Sample Count</label>
                      <input
                        type="number"
                        defaultValue="1000"
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-2">Privacy Budget (ε)</label>
                      <input
                        type="number"
                        step="0.1"
                        defaultValue="0.5"
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-400"
                      />
                    </div>
                  </div>

                  {isUploading && (
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Uploading with differential privacy...</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <div className="w-full bg-white/10 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-green-400 to-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  <button
                    onClick={uploadGradients}
                    disabled={isUploading}
                    className="w-full px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium flex items-center justify-center transition-all"
                  >
                    {isUploading ? (
                      <>
                        <Clock className="w-5 h-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5 mr-2" />
                        Submit Gradients
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Institutions Panel */}
        <div className="space-y-6">
          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold mb-4">Participating Institutions</h3>
            
            <div className="space-y-4">
              {institutions.map((institution) => (
                <div key={institution.id} className="bg-white/5 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium">{institution.name}</h4>
                    <span className={`text-xs ${getStatusColor(institution.status)}`}>
                      {institution.status}
                    </span>
                  </div>
                  
                  <div className="space-y-2 text-sm text-white/60">
                    <div className="flex justify-between">
                      <span>Contributions:</span>
                      <span>{institution.totalContributions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Privacy Used:</span>
                      <span>{institution.privacyBudgetUsed}/10.0</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Reputation:</span>
                      <span className="text-green-400">{(institution.reputationScore * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div className="mt-3">
                    <div className="w-full bg-white/10 rounded-full h-1">
                      <div 
                        className="bg-gradient-to-r from-blue-400 to-purple-500 h-1 rounded-full"
                        style={{ width: `${(institution.privacyBudgetUsed / 10) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold mb-4">Privacy Guarantees</h3>
            
            <div className="space-y-3 text-sm">
              <div className="flex items-start">
                <CheckCircle className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                <span>Differential privacy with ε-δ guarantees</span>
              </div>
              <div className="flex items-start">
                <CheckCircle className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                <span>Gradient clipping for bounded sensitivity</span>
              </div>
              <div className="flex items-start">
                <CheckCircle className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                <span>On-chain aggregation with threshold signatures</span>
              </div>
              <div className="flex items-start">
                <AlertTriangle className="w-4 h-4 text-yellow-400 mr-2 mt-0.5 flex-shrink-0" />
                <span>Privacy budget tracking per institution</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default FederatedLearning