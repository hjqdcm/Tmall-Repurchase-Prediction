package task1;

import java.io.Serializable;

public class BehaviorLog implements Serializable {
    private static final long serialVersionUID = 1L;
    
    public long userId;
    public long itemId;
    public int categoryId;
    public String behaviorType;
    public long timestamp;
    
    public BehaviorLog() {}
    
    public BehaviorLog(long userId, long itemId, int categoryId, 
                      String behaviorType, long timestamp) {
        this.userId = userId;
        this.itemId = itemId;
        this.categoryId = categoryId;
        this.behaviorType = behaviorType;
        this.timestamp = timestamp;
    }
    
    @Override
    public String toString() {
        return String.format("%d,%d,%d,%s,%d", 
            userId, itemId, categoryId, behaviorType, timestamp);
    }
}
