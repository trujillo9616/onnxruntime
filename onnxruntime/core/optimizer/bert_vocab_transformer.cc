// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/bert_vocab_transformer.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static Status SwapProducerConsumer(Graph& graph, Node* producer_node, Node* consumer_node, int producer_input_index) {
  graph_utils::ReplaceDownstreamNodeInput(graph, *consumer_node, 0, *producer_node, 0);
  graph_utils::ReplaceNodeInput(*consumer_node, 0, *producer_node->MutableInputDefs()[producer_input_index]);
  graph_utils::ReplaceNodeInput(*producer_node, producer_input_index, *consumer_node->MutableOutputDefs()[0]);
  consumer_node->MutableOutputDefs()[0]->ClearShape();
  producer_node->MutableOutputDefs()[0]->ClearShape();
  auto ret = graph.Resolve();
  ORT_ENFORCE(ret.IsOK());
  return Status::OK();
}

Status BertVocabTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const Node& next_node = *(node.OutputNodesBegin());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "SparseSoftmaxCrossEntropy", {9}, kOnnxDomain) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    bool stop = false;
    while (!stop) {
      //if (node.GetInputEdgesCount() > 0) {
      int index = 0;
      Node* input_node = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        std::cout << "Move GatherND" << node.Name() << " stopped at node " << input_node->Name() << std::endl;
        break;
      }

      // check the input node's output are not used by others
      if (input_node->OpType().compare("Add") == 0) {
        std::cout << "Move GatherND" << node.Name() << " up node node " << input_node->Name() << std::endl;

        if (graph_utils::IsGraphInput(graph, input_node->MutableInputDefs()[0]) ||
            graph_utils::IsInitializer(graph, input_node->MutableInputDefs()[0]->Name(), false)) {
          index = 1;
        }
        /* todo check second value be 128*/
        SwapProducerConsumer(graph, input_node, &node, index);
        modified = true;
      } else if (input_node->OpType().compare("MatMul") == 0) {
        // todo: check MatMul's first input's shape is [batch,seq, xx]
        std::cout << "Move GatherND" << node.Name() << " up node node " << input_node->Name() << std::endl;
        SwapProducerConsumer(graph, input_node, &node, index);
        modified = true;
      } else if (input_node->OpType().compare("LayerNormalization") == 0 || input_node->OpType().compare("Gelu") == 0) {
        // element-wise operators
        std::cout << "Move GatherND" << node.Name() << " up node node " << input_node->Name() << std::endl;
        SwapProducerConsumer(graph, input_node, &node, index);
        modified = true;
      } else {
        stop = true;
      }
      // } else {
      //   stop = true;
      // }
    }
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
    auto ret = graph.Resolve();
    ORT_ENFORCE(ret.IsOK());
  }
  return Status::OK();
}

}  // namespace onnxruntime
