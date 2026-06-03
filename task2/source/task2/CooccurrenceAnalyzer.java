package task2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

public class CooccurrenceAnalyzer {

    private static final String DEFAULT_DATA_BASE = "/user/root/final_exp/exp1";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf()
            .setAppName("Task2_CooccurrenceAnalysis");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        String hdfsUser = System.getenv("HADOOP_USER_NAME");
        if (hdfsUser == null || hdfsUser.isEmpty()) {
            hdfsUser = "231220053a";
        }

        String trainPath;
        String testPath;
        String outputBase;
        if (args.length >= 3) {
            trainPath = args[0];
            testPath = args[1];
            outputBase = args[2];
        } else if (args.length == 2) {
            // 本地单文件测试：args[0]=输入, args[1]=输出
            trainPath = args[0];
            testPath = args[0];
            outputBase = args[1];
        } else {
            trainPath = DEFAULT_DATA_BASE + "/train.csv";
            testPath = DEFAULT_DATA_BASE + "/test.csv";
            outputBase = "/user/" + hdfsUser + "/final_exp/exp2/output";
        }

        String outputIndexPath = outputBase + "/item_user_inverted_index.txt";
        String outputCooccurPath = outputBase + "/item_co_occurrence.txt";
        System.out.println("train:  " + trainPath);
        System.out.println("test:   " + testPath);
        System.out.println("output: " + outputBase);

        // train + test 合并处理
        JavaRDD<String> rawRdd = sc.textFile(trainPath).union(sc.textFile(testPath));

        JavaRDD<String> dataRdd = rawRdd.filter(line ->
            !line.startsWith("user_id") && !line.trim().isEmpty()
        );

        System.out.println("原始日志行数: " + dataRdd.count());

        JavaRDD<LogEntry> logsRdd = dataRdd
            .map(line -> {
                String[] parts = line.split(",");
                if (parts.length < 5) {
                    return null;
                }
                try {
                    return new LogEntry(
                        Long.parseLong(parts[0].trim()),
                        Long.parseLong(parts[1].trim()),
                        Integer.parseInt(parts[2].trim()),
                        parts[3].trim().toLowerCase(),
                        Long.parseLong(parts[4].trim())
                    );
                } catch (NumberFormatException e) {
                    return null;
                }
            })
            .filter(log -> log != null);

        JavaRDD<LogEntry> deepInteractRdd = logsRdd.filter(log ->
            log.behaviorType.equals("cart") ||
            log.behaviorType.equals("fav") ||
            log.behaviorType.equals("buy")
        );

        long deepCount = deepInteractRdd.count();
        System.out.println("深度交互记录数: " + deepCount);

        JavaPairRDD<Long, Iterable<Long>> invertedIndex = deepInteractRdd
            .mapToPair(log -> new Tuple2<>(log.itemId, log.userId))
            .distinct()
            .groupByKey()
            .mapValues(userIds -> {
                List<Long> list = new ArrayList<>();
                for (Long uid : userIds) {
                    list.add(uid);
                }
                Collections.sort(list);
                return list;
            });

        JavaRDD<String> invertedOutput = invertedIndex.map(tuple -> {
            String userIdsStr = new ArrayList<>(toList(tuple._2)).stream()
                .map(String::valueOf)
                .collect(Collectors.joining(","));
            return tuple._1 + "\t" + userIdsStr;
        });

        invertedOutput.saveAsTextFile(outputIndexPath);
        System.out.println("倒排索引已保存，商品数: " + invertedIndex.count());

        JavaPairRDD<Long, Iterable<Long>> userItemsRdd = deepInteractRdd
            .mapToPair(log -> new Tuple2<>(log.userId, log.itemId))
            .distinct()
            .groupByKey()
            .mapValues(items -> {
                List<Long> list = new ArrayList<>();
                for (Long item : items) {
                    list.add(item);
                }
                Collections.sort(list);
                return list;
            });

        System.out.println("有深度交互的用户数: " + userItemsRdd.count());

        JavaPairRDD<Tuple2<Long, Long>, Integer> cooccurRdd = userItemsRdd
            .flatMapToPair(tuple -> {
                List<Long> items = new ArrayList<>();
                for (Long item : tuple._2) {
                    items.add(item);
                }

                List<Tuple2<Tuple2<Long, Long>, Integer>> pairs = new ArrayList<>();
                int n = items.size();

                for (int i = 0; i < n; i++) {
                    for (int j = i + 1; j < n; j++) {
                        long itemA = items.get(i);
                        long itemB = items.get(j);
                        if (itemA < itemB) {
                            pairs.add(new Tuple2<>(new Tuple2<>(itemA, itemB), 1));
                        } else {
                            pairs.add(new Tuple2<>(new Tuple2<>(itemB, itemA), 1));
                        }
                    }
                }
                return pairs.iterator();
            })
            .reduceByKey(Integer::sum);

        long cooccurCount = cooccurRdd.count();
        System.out.println("共现对总数: " + cooccurCount);

        List<Tuple2<Tuple2<Long, Long>, Integer>> top1000 = cooccurRdd
            .mapToPair(t -> new Tuple2<>(t._2, t._1))
            .sortByKey(false)
            .mapToPair(t -> new Tuple2<>(t._2, t._1))
            .take(1000);

        System.out.println("取前1000条共现记录");

        JavaRDD<String> cooccurOutput = sc.parallelize(top1000).map(tuple -> {
            Tuple2<Long, Long> items = tuple._1;
            int count = tuple._2;
            return items._1 + "," + items._2 + "\t" + count;
        });

        cooccurOutput.coalesce(1).saveAsTextFile(outputCooccurPath);

        System.out.println("\n========== 任务二完成 ==========");
        System.out.println("深度交互记录数: " + deepCount);
        System.out.println("倒排索引商品数: " + invertedIndex.count());
        System.out.println("有交互的用户数: " + userItemsRdd.count());
        System.out.println("共现对总数: " + cooccurCount);
        System.out.println("前1000条已保存");

        System.out.println("\n共现结果示例 (前10条):");
        List<String> samples = cooccurOutput.take(10);
        for (String sample : samples) {
            System.out.println(sample);
        }

        sc.close();
    }

    private static List<Long> toList(Iterable<Long> iterable) {
        List<Long> list = new ArrayList<>();
        for (Long value : iterable) {
            list.add(value);
        }
        return list;
    }

    static class LogEntry implements Serializable {
        private static final long serialVersionUID = 1L;

        long userId;
        long itemId;
        int categoryId;
        String behaviorType;
        long timestamp;

        LogEntry(long userId, long itemId, int categoryId, String behaviorType, long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.categoryId = categoryId;
            this.behaviorType = behaviorType;
            this.timestamp = timestamp;
        }
    }
}
