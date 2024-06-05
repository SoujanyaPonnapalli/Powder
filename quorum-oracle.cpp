#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using std::vector;

void PrintProbabilities (int &numberOfReplicas, vector<double> &replicaFailureProbabilities) {
  std::cout << "Number of replicas: " << numberOfReplicas << std::endl;
  std::cout << "Probabilities:";
  for (int i = 0; i < numberOfReplicas; i++) {
    std::cout << " " << replicaFailureProbabilities[i];
  }
  std::cout << std::endl;
}

void EstimateProbabilitiesForNextView (
    int &numberOfReplicas, vector<double> &replicaFailureProbabilities,
    double defaultAdjustment = 0) {
    // double defaultAdjustment = 0.00005) {
  for (int i = 0; i < numberOfReplicas; i++) {
    replicaFailureProbabilities[i] += defaultAdjustment;
  }
  PrintProbabilities(numberOfReplicas, replicaFailureProbabilities);
}

void PredictFailureForConsensusRound (int &consensusRound, int &numberOfReplicas,
    vector<double> &replicaFailureProbabilities,
    double reqConsistency = 100) {
    // double reqConsistency = 99.99999) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> replicaDistribution(0, numberOfReplicas - 1);
  // Base case - I: Simple Quorums
  std::cout << std::endl << "NEW CONSENSUS ROUND = " << consensusRound << std::endl;
  for (int quorumSize = 1; quorumSize < numberOfReplicas; quorumSize++) {
    // Generate a distribution in the range 1 to numberOfReplicas
    // TODO: We need non-repetitive replicas (unique replicas)
    std::vector<int> estQuorum;
    double estReplicaFailure = 1.0;
    double estSimultaneousFailure = 1.0;

    for (int r = 0; r < quorumSize; r++) {
      int replicaID = replicaDistribution(rng);
      estQuorum.push_back(replicaID);
      estReplicaFailure = replicaFailureProbabilities[replicaID];
      estSimultaneousFailure *= estReplicaFailure;
    }
    std::cout << "QuorumSize = " << quorumSize << std::endl;
    std::cout << "Replicas in quorum";
    for (int r = 0; r < quorumSize; r++) {
      std::cout << " " << estQuorum[r];
    }
    std::cout << std::endl;
    PrintProbabilities(numberOfReplicas, replicaFailureProbabilities);
    if (estSimultaneousFailure <= 100 - reqConsistency) {
      std::cout << "Found A Quorum: " << estSimultaneousFailure << " vs " << 100 - reqConsistency << std::endl;
      break;
    }
    else {
      std::cout << "Couldn't Find A Quorum: " << estSimultaneousFailure << " vs " << 100 - reqConsistency << std::endl;
    }
  }
 
  // Base case - II: CFT Failure

  // Base case - III: BFT Failure

}

/*
** Input:
** (a) Number of replicas in the cluster
** (b) The probability of failure for each replica in the cluster
*/
int main (int argc, char **argv) {

  if (argc <= 2) {
    std::cout << "[ERROR] " << __func__ << ": Invalid Usage!" << std::endl;
    exit(-1);
  }
  int numberOfReplicas = atof(argv[1]);
  std::vector<double> replicaFailureProbabilities;
  for (int i = 0; i < numberOfReplicas; i++) {
    replicaFailureProbabilities.push_back(atof(argv[i + 2]));
  }
  PrintProbabilities(numberOfReplicas, replicaFailureProbabilities);
  for (int consensusRound = 0; consensusRound < 50; consensusRound++) {
    PredictFailureForConsensusRound(consensusRound, numberOfReplicas,replicaFailureProbabilities);
    EstimateProbabilitiesForNextView(numberOfReplicas, replicaFailureProbabilities);
  }
}



