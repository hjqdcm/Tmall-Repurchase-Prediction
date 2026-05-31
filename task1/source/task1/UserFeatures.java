package task1;

import java.io.Serializable;

public class UserFeatures implements Serializable {
    private static final long serialVersionUID = 1L;

    public long userId;
    /** F1: 距全局截止时间的时效差（天） */
    public double recencyDays;
    /** F2: 总交互次数 */
    public int activityCount;
    /** F3: 行为加权总分 */
    public double valueScore;

    public UserFeatures() {}

    public UserFeatures(long userId, double recencyDays, int activityCount, double valueScore) {
        this.userId = userId;
        this.recencyDays = recencyDays;
        this.activityCount = activityCount;
        this.valueScore = valueScore;
    }
}
