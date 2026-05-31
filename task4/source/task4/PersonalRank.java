package task4;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class PersonalRank {

    private static final double ALPHA = 0.85;
    private static final int MAX_ITERATIONS = 100;
    private static final double CONVERGENCE_THRESHOLD = 0.0001;
    private static final long TARGET_USER_ID = 1001L;
    private static final int TOP_K = 10;

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
            .setAppName("Task4_PersonalRank");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }
        String inputPath = args.length > 0 ? args[0]
            : "/user/root/final_exp/exp1/train.csv";
        String outputBase = args.length > 1 ? args[1]
            : "/user/" + hdfsUser + "/final_exp/exp4/output";
        String outputPath = outputBase + "/personal_rank_recommendations.txt";

        long targetUserId = TARGET_USER_ID;
        if (args.length > 2) {
            targetUserId = Long.parseLong(args[2]);
        }

        System.out.println("input:  " + inputPath);
        System.out.println("output: " + outputBase);
        System.out.println("目标用户: " + targetUserId);

        // ==================== 步骤1: 读取数据并过滤深度交互 (与任务2一致) ====================
        JavaRDD<String> rawRdd = sc.textFile(inputPath);

        JavaRDD<String> dataRdd = rawRdd.filter(line ->
            !line.startsWith("user_id") && !line.trim().isEmpty()
        );

        JavaRDD<LogEntry> deepInteractRdd = dataRdd
            .map(line -> {
                String[] parts = line.split(",");
                if (parts.length < 5) {
                    return null;
                }
                try {
                    return new LogEntry(
                        Long.parseLong(parts[0].trim()),
                        Long.parseLong(parts[1].trim()),
                        parts[3].trim().toLowerCase(),
                        Long.parseLong(parts[4].trim())
                    );
                } catch (NumberFormatException e) {
                    return null;
                }
            })
            .filter(log -> log != null)
            .filter(log -> log.behaviorType.equals("cart")
                || log.behaviorType.equals("fav")
                || log.behaviorType.equals("buy"));

        System.out.println("深度交互记录数: " + deepInteractRdd.count());

        // ==================== 步骤2: 构建 User-Item 无向二分图 ====================
        // 节点: U_userId / I_itemId；边: 双向 (U,I) 与 (I,U)
        JavaPairRDD<String, String> edges = deepInteractRdd
            .flatMapToPair(log -> {
                String userNode = "U_" + log.userId;
                String itemNode = "I_" + log.itemId;
                List<Tuple2<String, String>> edgeList = new ArrayList<>();
                edgeList.add(new Tuple2<>(userNode, itemNode));
                edgeList.add(new Tuple2<>(itemNode, userNode));
                return edgeList.iterator();
            })
            .distinct();

        JavaPairRDD<String, Integer> outDegree = edges
            .mapToPair(edge -> new Tuple2<>(edge._1, 1))
            .reduceByKey(Integer::sum);

        JavaRDD<String> allNodes = edges
            .flatMap(edge -> Arrays.asList(edge._1, edge._2).iterator())
            .distinct();

        long nodeCount = allNodes.count();
        System.out.println("总节点数: " + nodeCount);
        System.out.println("边数量: " + edges.count());

        Set<String> targetUserItems = getInteractedItems(targetUserId, deepInteractRdd);
        System.out.println("目标用户已交互商品数: " + targetUserItems.size());

        String targetNode = "U_" + targetUserId;
        List<String> nodeList = allNodes.collect();
        boolean targetInGraph = nodeList.contains(targetNode);
        if (!targetInGraph) {
            System.out.println("警告: 目标用户 " + targetUserId
                + " 在深度交互图中不存在，所有商品得分将为 0");
        } else if (targetUserItems.isEmpty()) {
            System.out.println("警告: 目标用户无 cart/fav/buy 记录，推荐结果可能全为 0");
        }

        // ==================== 步骤3: 初始化 PR 状态 (目标用户节点为 1，其余为 0) ====================
        JavaPairRDD<String, Double> prRdd = allNodes.mapToPair(node ->
            new Tuple2<>(node, targetNode.equals(node) ? 1.0 : 0.0));

        // ==================== 步骤4: join + reduceByKey 迭代更新 ====================
        // PR(v) = r(v)·((1-α) + α·danglingMass) + α·Σ_{u→v} PR(u)/outDegree(u)
        // r(v): 目标用户节点为 1，其余为 0；悬挂节点 PR 按 r(v) 重启
        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            System.out.println("迭代 " + (iter + 1) + "/" + MAX_ITERATIONS);

            JavaPairRDD<String, Tuple2<Double, Integer>> prWithOut = prRdd
                .leftOuterJoin(outDegree)
                .mapToPair(t -> new Tuple2<>(t._1,
                    new Tuple2<>(t._2._1, t._2._2.orElse(0))));

            double danglingMass = prWithOut
                .filter(t -> t._2._2 == 0)
                .map(t -> t._2._1)
                .fold(0.0, (a, b) -> a + b);

            JavaPairRDD<String, Tuple2<Double, Integer>> prActive = prWithOut
                .filter(t -> t._2._2 > 0);

            JavaPairRDD<String, Double> incoming = edges
                .join(prActive)
                .mapToPair(t -> new Tuple2<>(
                    t._2._1,
                    t._2._2._1 / t._2._2._2))
                .reduceByKey(Double::sum);

            final String restartNode = targetNode;
            final double finalDanglingMass = danglingMass;
            JavaPairRDD<String, Double> newPrRdd = allNodes
                .mapToPair(node -> new Tuple2<>(node, 0.0))
                .leftOuterJoin(incoming)
                .mapToPair(t -> {
                    String node = t._1;
                    double incomingSum = t._2._2.orElse(0.0);
                    double restartProb = node.equals(restartNode) ? 1.0 : 0.0;
                    double pr = restartProb * ((1.0 - ALPHA) + ALPHA * finalDanglingMass)
                        + ALPHA * incomingSum;
                    return new Tuple2<>(node, pr);
                });

            System.out.printf("  danglingMass: %.6f\n", finalDanglingMass);

            double maxChange = prRdd.join(newPrRdd)
                .values()
                .map(t -> Math.abs(t._1 - t._2))
                .reduce(Math::max);

            System.out.printf("  maxChange: %.6f\n", maxChange);

            prRdd = newPrRdd;

            if (maxChange < CONVERGENCE_THRESHOLD) {
                System.out.println("已收敛，停止迭代");
                break;
            }
        }

        // ==================== 步骤5: Top-K 推荐 (过滤已交互商品) ====================
        Map<String, Double> finalPrMap = prRdd.collectAsMap();

        List<Tuple2<String, Double>> itemScores = new ArrayList<>();
        for (Map.Entry<String, Double> entry : finalPrMap.entrySet()) {
            String node = entry.getKey();
            if (node.startsWith("I_")) {
                String itemId = node.substring(2);
                if (!targetUserItems.contains(itemId)) {
                    itemScores.add(new Tuple2<>(itemId, entry.getValue()));
                }
            }
        }

        itemScores.sort((a, b) -> {
            int cmp = Double.compare(b._2, a._2);
            return cmp != 0 ? cmp : a._1.compareTo(b._1);
        });
        List<Tuple2<String, Double>> topK = itemScores.subList(
            0, Math.min(TOP_K, itemScores.size()));

        double maxItemScore = topK.isEmpty() ? 0.0 : topK.get(0)._2;
        System.out.println("最高推荐得分: " + String.format("%.6f", maxItemScore));
        if (maxItemScore < 0.0005) {
            System.out.println("提示: 得分四舍五入到 3 位小数会显示为 0.000；"
                + "通常是目标用户所在连通分量过小，或该用户不在图中");
        }

        String outputLine = targetUserId + "\t" +
            topK.stream()
                .map(t -> t._1 + ":" + String.format("%.3f", t._2))
                .collect(Collectors.joining(","));

        sc.parallelize(Collections.singletonList(outputLine))
            .coalesce(1)
            .saveAsTextFile(outputPath);

        System.out.println("\n========== 任务四完成 ==========");
        System.out.println("目标用户: " + targetUserId);
        System.out.println("推荐结果: " + outputLine);
        System.out.println("已保存至: " + outputPath);

        System.out.println("\nTop " + TOP_K + " 推荐商品:");
        for (int i = 0; i < topK.size(); i++) {
            Tuple2<String, Double> item = topK.get(i);
            System.out.printf("  %d. 商品 %s, 得分: %.3f\n", i + 1, item._1, item._2);
        }

        sc.close();
    }

    private static Set<String> getInteractedItems(long targetUserId, JavaRDD<LogEntry> deepInteractRdd) {
        return deepInteractRdd
            .filter(log -> log.userId == targetUserId)
            .map(log -> String.valueOf(log.itemId))
            .distinct()
            .collect()
            .stream()
            .collect(Collectors.toSet());
    }

    static class LogEntry implements Serializable {
        long userId;
        long itemId;
        String behaviorType;
        long timestamp;

        LogEntry(long userId, long itemId, String behaviorType, long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.behaviorType = behaviorType;
            this.timestamp = timestamp;
        }
    }
}
