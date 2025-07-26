import React, { useState, useEffect } from 'react'
import { Brain, Shield, Network, Activity, Users, Lock, Zap, Database } from 'lucide-react'
import DiagnosisInterface from './components/DiagnosisInterface'
import FederatedLearning from './components/FederatedLearning'
import PrivacyDashboard from './components/PrivacyDashboard'
import BlockchainStatus from './components/BlockchainStatus'

function App() {
  const [activeTab, setActiveTab] = useState('diagnosis')
  const [connectionStatus, setConnectionStatus] = useState('disconnected')

  useEffect(() => {
    // Simulate connection to IC network
    setTimeout(() => {
      setConnectionStatus('connected')
    }, 2000)
  }, [])

  const tabs = [
    { id: 'diagnosis', label: 'AI Diagnosis', icon: Brain },
    { id: 'federated', label: 'Federated Learning', icon: Network },
    { id: 'privacy', label: 'Privacy Dashboard', icon: Shield },
    { id: 'blockchain', label: 'Blockchain Status', icon: Database },
  ]

  return (
    <div className="min-h-screen gradient-bg">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center mr-4">
                <Activity className="w-8 h-8 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-6 h-6 ai-gradient rounded-full flex items-center justify-center">
                <Brain className="w-3 h-3 text-white" />
              </div>
            </div>
            <h1 className="text-5xl font-bold text-white">
              MedChain AI
            </h1>
          </div>
          
          <p className="text-xl text-white/80 mb-4">
            Healthcare × AI Agents × Blockchain
          </p>
          
          <div className="flex items-center justify-center space-x-6 text-sm text-white/70">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                connectionStatus === 'connected' ? 'bg-green-400' : 'bg-yellow-400'
              }`} />
              {connectionStatus === 'connected' ? 'Connected to IC Network' : 'Connecting...'}
            </div>
            <div className="flex items-center">
              <Lock className="w-4 h-4 mr-1" />
              Threshold-ECDSA Enabled
            </div>
            <div className="flex items-center">
              <Shield className="w-4 h-4 mr-1" />
              Differential Privacy Active
            </div>
          </div>
        </header>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="glass-effect rounded-xl p-6 text-white">
            <div className="medical-gradient w-12 h-12 rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">On-Chain AI Inference</h3>
            <p className="text-white/70 text-sm">
              Rust canisters running AI models directly on blockchain with cryptographic guarantees
            </p>
          </div>
          
          <div className="glass-effect rounded-xl p-6 text-white">
            <div className="ai-gradient w-12 h-12 rounded-lg flex items-center justify-center mb-4">
              <Users className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Federated Learning</h3>
            <p className="text-white/70 text-sm">
              Privacy-preserving collaborative learning across medical institutions
            </p>
          </div>
          
          <div className="glass-effect rounded-xl p-6 text-white">
            <div className="blockchain-gradient w-12 h-12 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Differential Privacy</h3>
            <p className="text-white/70 text-sm">
              Mathematical privacy guarantees for medical data with on-chain gradient aggregation
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="glass-effect rounded-xl p-2 mb-8">
          <div className="flex space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center px-6 py-3 rounded-lg transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-white/20 text-white shadow-lg'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-5 h-5 mr-2" />
                  {tab.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Tab Content */}
        <div className="glass-effect rounded-xl p-8">
          {activeTab === 'diagnosis' && <DiagnosisInterface />}
          {activeTab === 'federated' && <FederatedLearning />}
          {activeTab === 'privacy' && <PrivacyDashboard />}
          {activeTab === 'blockchain' && <BlockchainStatus />}
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-white/60">
          <p className="text-sm">
            Building the first mainnet solution for differential-privacy gradient aggregation on-chain
          </p>
          <div className="flex items-center justify-center mt-4 space-x-4 text-xs">
            <span>Varion AI Integration</span>
            <span>•</span>
            <span>Threshold-ECDSA Security</span>
            <span>•</span>
            <span>Medical Data Privacy</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App