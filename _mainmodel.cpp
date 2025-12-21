
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <unordered_map>
#include <algorithm>
using namespace std;

struct csvData
{
    string label;
    int a, b, c, d;
};

// LABELING
int encodeLabel(const string &label)
{
    if (label == "L")
        return 0;
    if (label == "R")
        return 1;
    if (label == "B")
        return 2;
    return -1;
}
string reEncodeLabel(const int &label)
{
    if (label == 0)
        return "L";
    if (label == 1)
        return "R";
    if (label == 2)
        return "B";
    return "ERROR";
}

// DATA LOADING
vector<csvData> loadTrainData(const string &fileName)
{
    vector<csvData> dataSet;
    ifstream file(fileName);
    string line;

    if (!file.is_open())
    {
        std::filesystem::path alt = std::filesystem::path("decisiontree") / fileName;
        if (std::filesystem::exists(alt))
        {
            file.open(alt.string());
        }
    }
    if (!file.is_open())
    {
        cerr << "Unable to open training file '" << fileName << "' (cwd: " << std::filesystem::current_path() << ")" << endl;
        return dataSet;
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        csvData data;
        string temp;

        getline(ss, temp, ',');
        data.label = temp;
        getline(ss, temp, ',');
        data.a = stoi(temp);
        getline(ss, temp, ',');
        data.b = stoi(temp);
        getline(ss, temp, ',');
        data.c = stoi(temp);
        getline(ss, temp, ',');
        data.d = stoi(temp);

        dataSet.push_back(data);
    }

    return dataSet;
}

vector<csvData> loadTestData(const string &fileName)
{
    vector<csvData> dataSet;
    ifstream file(fileName);
    string line;

    if (!file.is_open())
    {
        std::filesystem::path alt = std::filesystem::path("decisiontree") / fileName;
        if (std::filesystem::exists(alt))
        {
            file.open(alt.string());
        }
    }
    if (!file.is_open())
    {
        cerr << "Unable to open test file '" << fileName << "' (cwd: " << std::filesystem::current_path() << ")" << endl;
        return dataSet;
    }

    while (getline(file, line))
    {
        istringstream ss(line);
        csvData data;
        string temp;

        getline(ss, temp, ',');
        data.a = stoi(temp);
        getline(ss, temp, ',');
        data.b = stoi(temp);
        getline(ss, temp, ',');
        data.c = stoi(temp);
        getline(ss, temp, ',');
        data.d = stoi(temp);

        dataSet.push_back(data);
    }

    return dataSet;
}

// TREE STRUCTURE
struct TreeNode
{
    int feature;
    int threshold;
    int label;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int f = -1, double t = 0, int l = 0)
        : feature(f), threshold(t), label(l), left(nullptr), right(nullptr) {}
};

// ENTROPY
double calcEntropy(const vector<csvData> &data)
{
    unordered_map<string, int> labelCounts;
    for (const auto &sample : data)
    {
        labelCounts[sample.label]++;
    }

    double entropy = 0.0;
    int totalSamples = data.size();
    for (const auto &[label, count] : labelCounts)
    {
        double probability = static_cast<double>(count) / totalSamples;
        entropy -= probability * log2(probability);
    }

    return entropy;
}

// SPLITING
pair<vector<csvData>, vector<csvData>> splitData(const vector<csvData> &data, int featureIndex, int threshold)
{
    vector<csvData> leftSplit, rightSplit;

    for (const auto &sample : data)
    {
        int value = (featureIndex == 0 ? sample.a : featureIndex == 1 ? sample.b
                                                : featureIndex == 2   ? sample.c
                                                                      : sample.d);
        if (value <= threshold)
            leftSplit.push_back(sample);
        else
            rightSplit.push_back(sample);
    }

    return {leftSplit, rightSplit};
}

// INFO GAIN
double calcInfoGain(const vector<csvData> &parent, const vector<csvData> &leftSplit, const vector<csvData> &rightSplit)
{
    double parentEntropy = calcEntropy(parent);
    double leftEntropy = calcEntropy(leftSplit);
    double rightEntropy = calcEntropy(rightSplit);

    int totalSize = parent.size();
    int leftSize = leftSplit.size();
    int rightSize = rightSplit.size();

    double weightedEntropy = (static_cast<double>(leftSize) / totalSize) * leftEntropy + (static_cast<double>(rightSize) / totalSize) * rightEntropy;

    return parentEntropy - weightedEntropy;
}

// FIND BEST SPLIT
pair<int, int> findBestSplit(const vector<csvData> &data)
{
    int bestFeature = -1;
    int bestThreshold = -1;
    double bestGain = -1.0;

    for (int featureIndex = 0; featureIndex < 4; ++featureIndex)
    {
        vector<int> featureValues;
        for (const auto &sample : data)
        {
            int value = (featureIndex == 0 ? sample.a : featureIndex == 1 ? sample.b
                                                    : featureIndex == 2   ? sample.c
                                                                          : sample.d);
            featureValues.push_back(value);
        }

        sort(featureValues.begin(), featureValues.end());
        featureValues.erase(unique(featureValues.begin(), featureValues.end()), featureValues.end());

        for (size_t i = 0; i < featureValues.size() - 1; ++i)
        {
            int threshold = (featureValues[i] + featureValues[i + 1]) / 2;

            auto [leftSplit, rightSplit] = splitData(data, featureIndex, threshold);

            if (leftSplit.empty() || rightSplit.empty())
                continue;

            double gain = calcInfoGain(data, leftSplit, rightSplit);

            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = featureIndex;
                bestThreshold = threshold;
            }
        }
    }

    return {bestFeature, bestThreshold};
}

// LABEL PURITY
bool isPure(const vector<csvData> &data)
{
    string firstLabel = data[0].label;
    for (const auto &sample : data)
    {
        if (sample.label != firstLabel)
            return false;
    }
    return true;
}

// MOST COMMON LABEL
string getMostCommonLabel(const vector<csvData> &data)
{
    unordered_map<string, int> labelCounts;
    for (const auto &sample : data)
    {
        labelCounts[sample.label]++;
    }

    string mostCommonLabel;
    int maxCount = 0;
    for (const auto &[label, count] : labelCounts)
    {
        if (count > maxCount)
        {
            maxCount = count;
            mostCommonLabel = label;
        }
    }

    return mostCommonLabel;
}

// BUILD TREE
TreeNode *buildTree(const vector<csvData> &data, int maxDepth, int maxLeafNodes, int currentDepth = 0,
                    int currentLeafCount = 0, int minSamplesSplit = 2, int minSamplesLeaf = 1)
{
    if (currentDepth >= maxDepth || isPure(data) || currentLeafCount >= maxLeafNodes)
    {
        return new TreeNode(-1, -1, encodeLabel(getMostCommonLabel(data)));
    }

    if (data.size() <= minSamplesSplit)
    {
        return new TreeNode(-1, -1, encodeLabel(getMostCommonLabel(data)));
    }

    auto [bestFeature, bestThreshold] = findBestSplit(data);

    if (bestFeature == -1)
    {
        return new TreeNode(-1, -1, encodeLabel(getMostCommonLabel(data)));
    }

    auto [leftSplit, rightSplit] = splitData(data, bestFeature, bestThreshold);

    if (leftSplit.size() < minSamplesLeaf || rightSplit.size() < minSamplesLeaf)
    {
        return new TreeNode(-1, -1, encodeLabel(getMostCommonLabel(data)));
    }

    TreeNode *root = new TreeNode(bestFeature, bestThreshold, -1);
    root->left = buildTree(leftSplit, maxDepth, maxLeafNodes, currentDepth + 1, currentLeafCount + 1, minSamplesSplit, minSamplesLeaf);
    root->right = buildTree(rightSplit, maxDepth, maxLeafNodes, currentDepth + 1, currentLeafCount + 1, minSamplesSplit, minSamplesLeaf);

    return root;
}

// PRINT TREE
void printTree(TreeNode *root, string indent = "")
{
    if (!root)
        return;

    if (root->label != -1)
    {
        cout << indent << "Leaf: " << reEncodeLabel(root->label) << endl;
        return;
    }

    cout << indent << "Feature: " << root->feature
         << ", Threshold: " << root->threshold << endl;

    cout << indent << "Left:" << endl;
    printTree(root->left, indent + "  ");

    cout << indent << "Right:" << endl;
    printTree(root->right, indent + "  ");
}

// SAVE TREE
void saveTree(TreeNode *root, ofstream &file)
{
    if (root == nullptr)
        return;

    if (root->label != -1)
    {
        file << "Leaf " << root->label << endl;
    }
    else
    {
        file << "Node " << root->feature << " " << root->threshold << endl;
        saveTree(root->left, file);
        saveTree(root->right, file);
    }
}

void saveDecisionTree(TreeNode *root, const string &filename)
{
    ofstream file(filename);
    if (file.is_open())
    {
        saveTree(root, file);
        file.close();
        cout << "Saved to " << filename << endl;
    }
    else
    {
        cout << "Unable to open file for saving" << endl;
    }
}

// LOAD TREE
TreeNode *loadTree(ifstream &file)
{
    string line;
    if (getline(file, line))
    {
        if (line.substr(0, 4) == "Leaf")
        {
            int label = stoi(line.substr(5));
            return new TreeNode(-1, -1, label);
        }
        else if (line.substr(0, 4) == "Node")
        {
            int feature, threshold;
            sscanf(line.c_str(), "Node %d %d", &feature, &threshold);
            TreeNode *node = new TreeNode(feature, threshold, -1);
            node->left = loadTree(file);
            node->right = loadTree(file);
            return node;
        }
    }
    return nullptr;
}

TreeNode *loadDecisionTree(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        std::filesystem::path alt = std::filesystem::path("decisiontree") / filename;
        if (std::filesystem::exists(alt))
        {
            file.open(alt.string());
        }
    }
    if (file.is_open())
    {
        TreeNode *root = loadTree(file);
        file.close();
        return root;
    }
    else
    {
        cerr << "Unable to open decision tree file '" << filename << "' (cwd: " << std::filesystem::current_path() << ")" << endl;
        return nullptr;
    }
}

// PREDICT
string predict(TreeNode *root, const csvData &sample)
{
    if (root->label != -1)
    {
        return root->label == 0 ? "L" : root->label == 1 ? "R"
                                                         : "B";
    }

    int value = (root->feature == 0 ? sample.a : root->feature == 1 ? sample.b
                                             : root->feature == 2   ? sample.c
                                                                    : sample.d);

    if (value <= root->threshold)
    {
        return predict(root->left, sample);
    }
    else
    {
        return predict(root->right, sample);
    }
}

// SAVE PREDICTION
void savePredictions(const vector<string> &predictions, const string &fileName)
{
    ofstream file(fileName);
    if (file.is_open())
    {
        for (int id = 1; id <= predictions.size(); ++id)
        {
            file << predictions[id - 1] << endl;
        }
        file.close();
        cout << "Predictions saved to " << fileName << endl;
    }
    else
    {
        cout << "Unable to open file for savingx predictions" << endl;
    }
}

int main()
{

    // vector<csvData> chosenDataSet = loadTrainData("train.csv");
    // TreeNode* decisionTree = buildTree(chosenDataSet, 7, 7, 0, 0, 15, 5);
    // saveDecisionTree(decisionTree, "treesave.txt");

    vector<csvData> chosenDataSet = loadTestData("test.csv");
    TreeNode *decisionTree = loadDecisionTree("treesave.txt");
    vector<string> predictions;
    for (const auto &sample : chosenDataSet)
    {
        string predicted = predict(decisionTree, sample);
        cout << predicted << endl;
        predictions.push_back(predict(decisionTree, sample));
    }
    savePredictions(predictions, "predict.txt");

    // printTree(decisionTree);

    return 0;
}
